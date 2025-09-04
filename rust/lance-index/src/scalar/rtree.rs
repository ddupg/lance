// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::scalar::btree::TrainingSource;
use crate::scalar::rtree::str_sort::STRSortExec;
use crate::scalar::{IndexStore, IndexWriter};
use arrow_array::{Array, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions, OneShotExec};
use num_traits::Bounded;
use snafu::location;
use std::sync::{Arc, LazyLock};

pub mod str_sort;

const DEFAULT_RTREE_PAGE_SIZE: u32 = 4096;
const RTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const RTREE_PAGES_NAME: &str = "page_data.lance";
const BATCH_SIZE_META_KEY: &str = "batch_size";

#[derive(Debug, Clone, PartialEq)]
pub struct RTreeMetadata {
    pub(crate) page_size: u32,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_bounds: Vec<usize>,
}

impl RTreeMetadata {
    pub fn new(num_items: usize, page_size: u32) -> Self {
        let (num_nodes, level_bounds) = Self::compute_num_nodes(num_items, page_size);

        Self {
            page_size,
            num_items,
            num_nodes,
            level_bounds,
        }
    }

    /// Calculate the total number of nodes in the R-tree to allocate space for    
    /// and the index of each tree level (used in search later)
    pub(crate) fn compute_num_nodes(num_items: usize, node_size: u32) -> (usize, Vec<usize>) {
        let mut n = num_items;
        let mut num_nodes = n;
        let mut level_bounds = vec![n];
        while n > 1 {
            n = (n as f64 / node_size as f64).ceil() as usize;
            num_nodes += n;
            level_bounds.push(num_nodes);
        }
        (num_nodes, level_bounds)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Eq for BoundingBox {}

impl BoundingBox {
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn update(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        self.min_x = self.min_x.min(min_x);
        self.min_y = self.min_y.min(min_y);
        self.max_x = self.max_x.max(max_x);
        self.max_y = self.max_y.max(max_y);
    }

    fn update_with_bbox(&mut self, other: &BoundingBox) {
        self.update(other.min_x, other.min_y, other.max_x, other.max_y);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RTree {
    pub(crate) buffer: Vec<BoundingBox>,
    pub(crate) metadata: RTreeMetadata,
}

/// Extract bounding boxes from geometry columns
///
/// This function handles struct<x: double, y: double> format for point data
/// and extracts bounding boxes for spatial indexing.
pub fn extract_bounding_boxes(
    geometry_field: &Field,
    geometry_array: &dyn Array,
) -> Result<Vec<BoundingBox>> {
    use arrow_schema::DataType;

    match geometry_field.data_type() {
        DataType::Struct(fields) => {
            // Check if it's a struct with x, y fields (point data)
            let field_names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();

            if field_names.len() == 2 && field_names.contains(&"x") && field_names.contains(&"y") {
                extract_bboxes_from_point_struct(geometry_array)
            } else {
                Err(Error::Index {
                    message: format!("Unsupported struct format. Expected struct with x,y fields, got fields: {:?}", field_names),
                    location: snafu::location!(),
                })
            }
        }
        _ => Err(Error::Index {
            message: format!(
                "Unsupported geometry data type: {:?}. Expected struct<x: double, y: double>",
                geometry_field.data_type()
            ),
            location: snafu::location!(),
        }),
    }
}

fn extract_bboxes_from_point_struct(array: &dyn Array) -> Result<Vec<BoundingBox>> {
    use arrow_array::StructArray;

    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::Index {
            message: "Expected StructArray for point data".to_string(),
            location: snafu::location!(),
        })?;

    // Get x and y columns from the struct
    let x_column = struct_array
        .column_by_name("x")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'x' field".to_string(),
            location: snafu::location!(),
        })?;

    let y_column = struct_array
        .column_by_name("y")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'y' field".to_string(),
            location: snafu::location!(),
        })?;

    extract_point_bboxes(x_column.as_ref(), y_column.as_ref())
}

fn extract_point_bboxes(x_array: &dyn Array, y_array: &dyn Array) -> Result<Vec<BoundingBox>> {
    use arrow_array::{Float32Array, Float64Array};

    let len = x_array.len().min(y_array.len());
    let mut bboxes = Vec::with_capacity(len);

    // Handle Float64 coordinates
    if let (Some(x_f64), Some(y_f64)) = (
        x_array.as_any().downcast_ref::<Float64Array>(),
        y_array.as_any().downcast_ref::<Float64Array>(),
    ) {
        for i in 0..len {
            if x_f64.is_null(i) || y_f64.is_null(i) {
                continue;
            }

            let x = x_f64.value(i);
            let y = y_f64.value(i);

            // For points, bounding box is just the point itself
            bboxes.push(BoundingBox::new(x, y, x, y));
        }
    }
    // Handle Float32 coordinates
    else if let (Some(x_f32), Some(y_f32)) = (
        x_array.as_any().downcast_ref::<Float32Array>(),
        y_array.as_any().downcast_ref::<Float32Array>(),
    ) {
        for i in 0..len {
            if x_f32.is_null(i) || y_f32.is_null(i) {
                continue;
            }

            let x = x_f32.value(i) as f64;
            let y = y_f32.value(i) as f64;

            bboxes.push(BoundingBox::new(x, y, x, y));
        }
    } else {
        return Err(Error::Index {
            message: "Unsupported coordinate data type (expected Float32 or Float64)".to_string(),
            location: snafu::location!(),
        });
    }

    Ok(bboxes)
}

/// Train a btree index from a stream of sorted page-size batches of values and row ids
///
/// Note: This is likely to change.  It is unreasonable to expect the caller to do the sorting
/// and re-chunking into page-size batches.  This is left for simplicity as this feature is still
/// a work in progress
pub async fn train_rtree_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let data_source = Box::new(RTreeDataSource::new(data_source));

    let mut batch_stream = data_source
        .scan_ordered_chunks(DEFAULT_RTREE_PAGE_SIZE)
        .await?;

    // TODO: write data/metadata
    Ok(())
}

pub static BBOX_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("min_x", DataType::Float64, false),
        Field::new("min_y", DataType::Float64, false),
        Field::new("max_x", DataType::Float64, false),
        Field::new("max_y", DataType::Float64, false),
        Field::new("_rowid", DataType::UInt64, true),
    ]))
});

struct RTreeDataSource {
    source: Box<dyn TrainingSource>,
}

impl RTreeDataSource {
    fn new(source: Box<dyn TrainingSource>) -> Self {
        Self { source }
    }

    fn convert_bbox_stream(source: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
        // 创建一个流，将原始数据转换为边界框
        let bbox_stream = source
            .map_err(DataFusionError::into)
            .and_then(move |batch| async move {
                let schema = batch.schema();
                let geometry_field = schema.field(0);
                let geometry_array = batch.column(0);

                // 提取边界框
                let bboxes = extract_bounding_boxes(geometry_field, geometry_array.as_ref())?;

                // 创建边界框字段的数组
                let min_x_values: Vec<f64> = bboxes.iter().map(|bbox| bbox.min_x).collect();
                let min_y_values: Vec<f64> = bboxes.iter().map(|bbox| bbox.min_y).collect();
                let max_x_values: Vec<f64> = bboxes.iter().map(|bbox| bbox.max_x).collect();
                let max_y_values: Vec<f64> = bboxes.iter().map(|bbox| bbox.max_y).collect();

                // 获取row_id列
                let row_ids = batch.column(1).clone();

                // 创建新的记录批次
                RecordBatch::try_new(
                    BBOX_SCHEMA.clone(),
                    vec![
                        Arc::new(Float64Array::from(min_x_values)),
                        Arc::new(Float64Array::from(min_y_values)),
                        Arc::new(Float64Array::from(max_x_values)),
                        Arc::new(Float64Array::from(max_y_values)),
                        row_ids,
                    ],
                )
                .map_err(DataFusionError::from)
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            BBOX_SCHEMA.clone(),
            bbox_stream,
        )))
    }
}

#[async_trait]
impl TrainingSource for RTreeDataSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        // 读取原始数据
        let stream = self.source.scan_unordered_chunks(chunk_size).await?;

        let bbox_stream = Self::convert_bbox_stream(stream)?;

        // 使用RecordBatchStreamAdapter创建符合RecordBatchStream要求的流
        let input = Arc::new(OneShotExec::new(bbox_stream));

        // 执行排序计划
        let sorted_stream = execute_plan(
            Arc::new(STRSortExec::try_new(input, chunk_size, 1 << 30)?),
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )?;

        // 返回排序后的流
        Ok(sorted_stream)
    }

    async fn scan_unordered_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!()
    }

    async fn scan_aligned_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!()
    }
}

struct EncodedBatch {
    bbox: BoundingBox,
    page_number: u32,
}

async fn train_rtree_page(
    batch: RecordBatch,
    batch_idx: u32,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedBatch> {
    let bbox = analyze_batch(&batch)?;
    writer.write_record_batch(batch).await?;
    Ok(EncodedBatch {
        bbox,
        page_number: batch_idx,
    })
}

fn analyze_batch(batch: &RecordBatch) -> Result<BoundingBox> {
    let mut bbox = BoundingBox::new(
        f64::max_value(),
        f64::max_value(),
        f64::min_value(),
        f64::min_value(),
    );
    let column_mapping = |index| batch.column(index).as_any().downcast_ref::<Float64Array>();
    let min_x = column_mapping(0);
    let min_y = column_mapping(1);
    let max_x = column_mapping(2);
    let max_y = column_mapping(3);
    let (min_x, min_y, max_x, max_y) = match (min_x, min_y, max_x, max_y) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return Err(Error::Arrow {
            message: "BBOX_SCHEMA must be Float64 columns (min_x,min_y,max_x,max_y)".into(),
            location: location!(),
        }),
    };
    let len = min_x.len();
    for i in 0..len {
        bbox.update(min_x.value(i), min_y.value(i), max_x.value(i), max_y.value(i));
    }
    Ok(bbox)
}
