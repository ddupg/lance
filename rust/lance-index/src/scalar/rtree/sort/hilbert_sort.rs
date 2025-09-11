// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::scalar::rtree::{BoundingBox, BBOX_ROWID_SCHEMA};
use crate::scalar::{IndexReaderStream, IndexStore};
use crate::Result;
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{
    DataType as ArrowDataType, DataType, Field as ArrowField,
};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{ColumnarValue, Signature, Volatility};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_physical_expr::expressions::{Column as DFColumn};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use futures::{StreamExt, TryFutureExt};
use lance_core::Error;
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions, OneShotExec};
use num_traits::Bounded;
use snafu::location;
use std::any::Any;
use std::sync::{Arc, LazyLock};
use async_trait::async_trait;
use crate::scalar::rtree::sort::Sorter;

const HILBERT_FIELD_NAME: &str = "_hilbert";

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

    fn tmp_spill_filename(&self) -> &str {
        "spill.lance.tmp"
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
                message: "bbox must be Float64 columns (min_x,min_y,max_x,max_y)"
                    .to_string(),
                location: location!(),
            }),
        }
    }
}

#[async_trait]
impl Sorter for HilbertSorter {
    async fn sort(
        &self,
        mut data: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        // 1. Scan source data statistics bbox, and spill data to disk
        let mut writer = self
            .spill_store
            .new_index_file(self.tmp_spill_filename(), BBOX_ROWID_SCHEMA.clone())
            .await?;

        let mut bbox = BoundingBox::new(
            f64::max_value(),
            f64::max_value(),
            f64::min_value(),
            f64::min_value(),
        );
        while let Some(batch) = data.next().await {
            let batch = batch?;
            let (min_x, min_y, max_x, max_y) = Self::extract_coord_array(&batch)?;
            for i in 0.. min_x.len() {
                bbox.update(
                    min_x.value(i),
                    min_y.value(i),
                    max_x.value(i),
                    max_y.value(i),
                );
            }
            writer.write_record_batch(batch).await?;
        }
        writer.finish().await?;

        // 第二阶段：使用DataFusion ExecutionPlan处理数据
        // 1. 创建数据源
        let reader = self
            .spill_store
            .open_index_file(self.tmp_spill_filename())
            .await?;
        let stream = IndexReaderStream::new(reader, self.page_size as u64)
            .await
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(self.spill_store.io_parallelism())
            .boxed();
        let source = Arc::new(OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
            BBOX_ROWID_SCHEMA.clone(),
            stream,
        ))));

        // 2. Add _hilbert column
        let mut projection_exprs = BBOX_ROWID_SCHEMA
            .fields()
            .iter()
            .map(|f| f.name())
            .enumerate()
            .map(|(idx, field_name)| {
                (
                    Arc::new(DFColumn::new(field_name, idx)) as Arc<dyn PhysicalExpr>,
                    field_name.to_string(),
                )
            })
            .collect::<Vec<_>>();
        projection_exprs.push((
            HilbertUDF::new(bbox).into_physical_expr(),
            HILBERT_FIELD_NAME.to_string(),
        ));

        let projection = Arc::new(ProjectionExec::try_new(
            projection_exprs,
            source as Arc<dyn ExecutionPlan>,
        )?);

        // 3. sort_by _hilbert
        let sort_expr = PhysicalSortExpr {
            expr: Arc::new(DFColumn::new(HILBERT_FIELD_NAME, 6)), // _hilbert column
            options: arrow_schema::SortOptions::default(),
        };

        let sort_exec = Arc::new(SortExec::new(
            [sort_expr].into(),
            projection as Arc<dyn ExecutionPlan>,
        ));

        let sorted_stream = execute_plan(
            sort_exec,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )?;

        // 4. clean temp file
        self.spill_store
            .delete_index_file(self.tmp_spill_filename())
            .await?;

        Ok(sorted_stream)
    }

    async fn cleanup(&self) -> Result<()> {
        self.spill_store.delete_index_file(self.tmp_spill_filename()).await
    }
}

const HILBERT_UDF_NAME: &str = "hilbert";

#[derive(Debug, Clone)]
struct HilbertUDF {
    signature: Signature,
    bbox: BoundingBox,
    width: f64,
    height: f64,
}

impl HilbertUDF {
    fn new(bbox: BoundingBox) -> Self {
        let signature = Signature::exact(
            vec![
                DataType::Float64, // min_x
                DataType::Float64, // min_y
                DataType::Float64, // max_x
                DataType::Float64, // max_y
            ],
            Volatility::Immutable,
        );
        let width = bbox.max_x - bbox.min_x;
        let height = bbox.max_y - bbox.min_y;
        Self {
            signature,
            bbox,
            width,
            height,
        }
    }

    fn into_physical_expr(self) -> Arc<dyn PhysicalExpr> {
        Arc::new(ScalarFunctionExpr::new(
            HILBERT_UDF_NAME,
            Arc::new(self.into()),
            vec![
                Arc::new(DFColumn::new("min_x", 0)) as Arc<dyn PhysicalExpr>,
                Arc::new(DFColumn::new("min_y", 1)) as Arc<dyn PhysicalExpr>,
                Arc::new(DFColumn::new("max_x", 2)) as Arc<dyn PhysicalExpr>,
                Arc::new(DFColumn::new("max_y", 3)) as Arc<dyn PhysicalExpr>,
            ],
            Arc::new(ArrowField::new(
                HILBERT_FIELD_NAME,
                ArrowDataType::Float64,
                false,
            )),
        ))
    }
}

impl ScalarUDFImpl for HilbertUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        HILBERT_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[ArrowDataType]) -> DataFusionResult<ArrowDataType> {
        Ok(ArrowDataType::UInt32)
    }

    fn invoke_with_args(&self, func_args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let arg_arrays = func_args.args[..4]
            .iter()
            .map(|arg| {
                let array = match arg {
                    ColumnarValue::Array(array) => array,
                    _ => Err(DataFusionError::Execution(
                        "hilbert only supports array arguments".to_owned(),
                    ))?,
                };

                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "hilbert only supports Float64 arguments".to_owned(),
                        )
                    })
            })
            .collect::<DataFusionResult<Vec<&Float64Array>>>()?;

        let min_x_arr = &arg_arrays[0];
        let min_y_arr = &arg_arrays[1];
        let max_x_arr = &arg_arrays[2];
        let max_y_arr = &arg_arrays[3];

        let hilbert_max = ((1 << 16) - 1) as f64;
        let len = min_x_arr.len();
        let mut hilbert_values = Vec::with_capacity(len);
        for i in 0..len {
            let min_x = min_x_arr.value(i);
            let min_y = min_y_arr.value(i);
            let max_x = max_x_arr.value(i);
            let max_y = max_y_arr.value(i);

            let x = (hilbert_max * ((min_x + max_x) / 2. - self.bbox.min_x) / self.width).floor()
                as u32;
            let y = (hilbert_max * ((min_y + max_y) / 2. - self.bbox.min_y) / self.height).floor()
                as u32;
            hilbert_values.push(hilbert_curve(x, y));
        }

        Ok(ColumnarValue::Array(
            Arc::new(UInt32Array::from(hilbert_values)) as ArrayRef,
        ))
    }
}

/// Fast Hilbert curve algorithm by http://threadlocalmutex.com/
/// Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
#[inline]
fn hilbert_curve(x: u32, y: u32) -> u32 {
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
