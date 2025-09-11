// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::scalar::rtree::{BoundingBox, BBOX_SCHEMA};
use crate::scalar::{IndexReaderStream, IndexStore};
use crate::Result;
use arrow_array::{Array, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, Schema,
};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_physical_expr::expressions::Column;
use futures::{FutureExt, StreamExt, TryFutureExt};
use lance_core::Error;
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions, OneShotExec};
use num_traits::Bounded;
use snafu::location;
use std::sync::{Arc, LazyLock};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;

const HILBERT_FIELD_NAME: &str = "_hilbert";
pub static BBOX_HILBERT_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    let mut fields = BBOX_SCHEMA.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(
        HILBERT_FIELD_NAME,
        ArrowDataType::Float64,
        false,
    )));
    Arc::new(ArrowSchema::new(fields))
});

pub(crate) struct HilbertSorter {
    page_size: u32,
    spill_store: Arc<dyn IndexStore>,
}

impl HilbertSorter {
    pub fn new(page_size: u32, spill_store: Arc<dyn IndexStore>) -> Self {
        Self {
            page_size,
            spill_store,
        }
    }

    fn tmp_spill_data_filename(&self) -> &str {
        "spill-data.lance.tmp"
    }

    fn tmp_spill_hilbert_filename(&self) -> &str {
        "spill-hilbert.lance.tmp"
    }

    fn extract_coord_array(
        batch: &RecordBatch,
    ) -> Result<(&Float64Array, &Float64Array, &Float64Array, &Float64Array)> {
        let column_mapping = |index| batch.column(index).as_any().downcast_ref::<Float64Array>();
        let min_x = column_mapping(0);
        let min_y = column_mapping(1);
        let max_x = column_mapping(2);
        let max_y = column_mapping(3);
        match (min_x, min_y, max_x, max_y) {
            (Some(a), Some(b), Some(c), Some(d)) => Ok((a, b, c, d)),
            _ => Err(Error::Index {
                message: "BBOX_SCHEMA must be Float64 columns (min_x,min_y,max_x,max_y)"
                    .to_string(),
                location: location!(),
            }),
        }
    }

    pub(crate) async fn sort(
        &self,
        mut data: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let mut writer = self
            .spill_store
            .new_index_file(self.tmp_spill_data_filename(), BBOX_SCHEMA.clone())
            .await?;

        let mut bbox = BoundingBox::new(
            f64::max_value(),
            f64::max_value(),
            f64::min_value(),
            f64::min_value(),
        );
        let mut num_items = 0;
        while let Some(batch) = data.next().await {
            let batch = batch?;
            let (min_x, min_y, max_x, max_y) = Self::extract_coord_array(&batch)?;
            let len = min_x.len();
            for i in 0..len {
                bbox.update(
                    min_x.value(i),
                    min_y.value(i),
                    max_x.value(i),
                    max_y.value(i),
                );
            }
            num_items += len;
            writer.write_record_batch(batch).await?;
        }
        writer.finish().await?;

        let width = bbox.max_x - bbox.min_x; // || 1.0;
        let height = bbox.max_y - bbox.min_y; // || 1.0;
        let hilbert_max = ((1 << 16) - 1) as f64;

        let mut writer = self
            .spill_store
            .new_index_file(
                self.tmp_spill_hilbert_filename(),
                BBOX_HILBERT_SCHEMA.clone(),
            )
            .await?;

        let mut reader_stream = IndexReaderStream::new(
            self.spill_store
                .open_index_file(self.tmp_spill_data_filename())
                .await?,
            self.page_size as u64,
        )
        .await;
        while let Some(batch) = reader_stream.next().await {
            let batch = batch.await?;
            let (min_x, min_y, max_x, max_y) = Self::extract_coord_array(&batch)?;
            let hilbert_arr = UInt32Array::from_iter_values((0..min_x.len()).map(|i| {
                let min_x = min_x.value(i);
                let min_y = min_y.value(i);
                let max_x = max_x.value(i);
                let max_y = max_y.value(i);
                let x = (hilbert_max * ((min_x + max_x) / 2. - bbox.min_x) / width).floor() as u32;
                let y = (hilbert_max * ((min_y + max_y) / 2. - bbox.min_y) / height).floor() as u32;
                hilbert(x, y)
            }));
            let mut cols = batch.columns().to_vec();
            cols.push(Arc::new(hilbert_arr));
            let augmented = match RecordBatch::try_new(BBOX_HILBERT_SCHEMA.clone(), cols) {
                Ok(b) => b,
                Err(e) => return Err(Error::from(e)),
            };
            writer.write_record_batch(augmented).await?;
        }
        writer.finish().await?;
        self.spill_store
            .delete_index_file(self.tmp_spill_data_filename())
            .await?;

        let sorted_by_hilbert_stream: SendableRecordBatchStream = {
            let reader = self
                .spill_store
                .open_index_file(self.tmp_spill_hilbert_filename())
                .await?;
            let stream = IndexReaderStream::new(reader, self.page_size as u64).await
                .map(|fut| fut.map_err(DataFusionError::from))
                .boxed();
            let stream = RecordBatchStreamAdapter::new(BBOX_HILBERT_SCHEMA.clone(), stream);
            let source = Arc::new(OneShotExec::new(Box::pin(stream)));
            let key_index = BBOX_HILBERT_SCHEMA.fields().len() - 1; // _hilbert
            let sort_expr = PhysicalSortExpr {
                expr: Arc::new(Column::new(HILBERT_FIELD_NAME, key_index)),
                options: arrow_schema::SortOptions::default(),
            };
            execute_plan(
                Arc::new(SortExec::new(
                    [sort_expr].into(),
                    source as Arc<dyn ExecutionPlan>,
                )),
                LanceExecutionOptions {
                    use_spilling: true,
                    ..Default::default()
                },
            )?
        };
        Ok(sorted_by_hilbert_stream)
    }
}

// Taken from static_aabb2d_index under the mit/apache license
// https://github.com/jbuckmccready/static_aabb2d_index/blob/9e6add59d77b74d4de0ac32159db47fbcb3acc28/src/static_aabb2d_index.rs#L486C1-L544C2
#[inline]
fn hilbert(x: u32, y: u32) -> u32 {
    // Fast Hilbert curve algorithm by http://threadlocalmutex.com/
    // Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF_00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F_0F0F;
    i0 = (i0 | (i0 << 2)) & 0x3333_3333;
    i0 = (i0 | (i0 << 1)) & 0x5555_5555;

    i1 = (i1 | (i1 << 8)) & 0x00FF_00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F_0F0F;
    i1 = (i1 | (i1 << 2)) & 0x3333_3333;
    i1 = (i1 | (i1 << 1)) & 0x5555_5555;

    (i1 << 1) | i0
}
