// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{CreatedIndex, IndexStore, IndexWriter, ScalarIndex};
use arrow_array::{new_null_array, Array, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::{Error, Result, ROW_ID};
use lance_io::object_store::ObjectStore;
use num_traits::Bounded;
use object_store::path::Path;
use snafu::location;
use sort::hilbert_sort::HilbertSorter;
use std::sync::{Arc, LazyLock};
use tempfile::{tempdir, TempDir};

mod sort;

const DEFAULT_RTREE_PAGE_SIZE: u32 = 4096;
const RTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const RTREE_PAGES_NAME: &str = "page_data.lance";
const BATCH_SIZE_META_KEY: &str = "batch_size";
const RTREE_INDEX_VERSION: u32 = 0;

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
    geometry_field: &ArrowField,
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
    data_source: SendableRecordBatchStream,
    index_store: &dyn IndexStore,
) -> Result<()> {
    // TODO: write data/metadata
    Ok(())
}

pub static BBOX_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("min_x", DataType::Float64, false),
        ArrowField::new("min_y", DataType::Float64, false),
        ArrowField::new("max_x", DataType::Float64, false),
        ArrowField::new("max_y", DataType::Float64, false),
    ]))
});
pub static BBOX_ROWID_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    let mut fields = BBOX_SCHEMA.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(ROW_ID, DataType::UInt64, true)));
    Arc::new(ArrowSchema::new(fields))
});
const PAGE_ID_COLUMN: &str = "_page_id";
pub static RTREE_PAGE_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    let mut fields = BBOX_ROWID_SCHEMA
        .fields()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(
        PAGE_ID_COLUMN,
        DataType::UInt64,
        true,
    )));
    Arc::new(ArrowSchema::new(fields))
});

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
                BBOX_ROWID_SCHEMA.clone(),
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
        BBOX_ROWID_SCHEMA.clone(),
        bbox_stream,
    )))
}

// #[async_trait]
// impl RTreeDataSource {
//     async fn scan_ordered_chunks(
//         self: Box<Self>,
//         chunk_size: u32,
//     ) -> Result<SendableRecordBatchStream> {
//         // 读取原始数据
//         let stream = self.source.scan_unordered_chunks(chunk_size).await?;
//
//         let bbox_stream = Self::convert_bbox_stream(stream)?;
//
//         // 使用RecordBatchStreamAdapter创建符合RecordBatchStream要求的流
//         let input = Arc::new(OneShotExec::new(bbox_stream));
//
//         // 执行排序计划
//         let sorted_stream = execute_plan(
//             Arc::new(STRSortExec::try_new(input, chunk_size, 1 << 30)?),
//             LanceExecutionOptions {
//                 use_spilling: true,
//                 ..Default::default()
//             },
//         )?;
//
//         // 返回排序后的流
//         Ok(sorted_stream)
//     }
// }

#[derive(Debug, Clone)]
pub struct RTreeIndexBuilderOptions {}

impl Default for RTreeIndexBuilderOptions {
    fn default() -> RTreeIndexBuilderOptions {
        RTreeIndexBuilderOptions {}
    }
}

pub struct RTreeIndexBuilder {
    options: RTreeIndexBuilderOptions,
    tmpdir: Arc<TempDir>,
    spill_store: Arc<dyn IndexStore>,
}

impl RTreeIndexBuilder {
    pub fn try_new(options: RTreeIndexBuilderOptions) -> Result<Self> {
        let tmpdir = Arc::new(tempdir()?);
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path())?,
            Arc::new(LanceCache::no_cache()),
        ));

        Ok(Self {
            options,
            tmpdir,
            spill_store,
        })
    }

    fn validate_value_field(field: &ArrowField) -> Result<()> {
        field
            .metadata()
            .get("ARROW:extension:name")
            .filter(|v| v.starts_with("geoarrow."))
            .ok_or_else(|| Error::InvalidInput {
                source: "Field must have a geoarrow extension type".into(),
                location: location!(),
            })?;
        Ok(())
    }

    fn validate_schema(schema: &ArrowSchema) -> Result<()> {
        if schema.fields().len() != 2 {
            return Err(Error::InvalidInput {
                source: "RTree index schema must have exactly two fields".into(),
                location: location!(),
            });
        }
        let values_field = schema.field_with_name(VALUE_COLUMN_NAME)?;
        Self::validate_value_field(values_field)?;

        let row_id_field = schema.field_with_name(ROW_ID)?;
        if *row_id_field.data_type() != DataType::UInt64 {
            return Err(Error::InvalidInput {
                source: "Second field in RTree index schema must be of type UInt64".into(),
                location: location!(),
            });
        }
        Ok(())
    }

    pub async fn train(
        &mut self,
        data: SendableRecordBatchStream,
        store: &dyn IndexStore,
    ) -> Result<()> {
        let schema = data.schema();
        Self::validate_schema(schema.as_ref())?;

        let bbox_data = convert_bbox_stream(data)?;
        // new sorted stream
        let sorter = HilbertSorter::new(DEFAULT_RTREE_PAGE_SIZE, self.spill_store.clone());
        let sorted_data = sorter.sort(bbox_data).await?;

        self.write_index(sorted_data, store, DEFAULT_RTREE_PAGE_SIZE).await?;

        Ok(())
    }

    pub async fn write_index(
        &mut self,
        sorted_data: SendableRecordBatchStream,
        store: &dyn IndexStore,
        page_size: u32,
    ) -> Result<()> {
        let mut batch_idx: u32 = 0;
        let mut writer = store
            .new_index_file(RTREE_PAGES_NAME, RTREE_PAGE_SCHEMA.clone())
            .await?;

        let mut current_level = Some(sorted_data);
        while let Some(mut data) = current_level.take() {
            let mut next_level = vec![];
            while let Some(batch) = data.next().await {
                let batch = batch?;
                let encoded_batch = train_rtree_page(batch, batch_idx, writer.as_mut()).await?;
                batch_idx += 1;
                next_level.push(encoded_batch);
            }

            if !next_level.is_empty() {
                current_level = Some(EncodedBatch::batches_into_batch_stream(
                    next_level, page_size,
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct RTreeIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for RTreeIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        field: &ArrowField,
    ) -> Result<Box<dyn TrainingRequest>> {
        // Check if field has geoarrow extension type
        RTreeIndexBuilder::validate_value_field(field)?;

        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        )))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
    ) -> Result<CreatedIndex> {
        let mut builder = RTreeIndexBuilder::try_new(RTreeIndexBuilderOptions::default())?;
        builder.train(data, index_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::RTreeIndexDetails::default())?,
            index_version: RTREE_INDEX_VERSION,
        })
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        RTREE_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        todo!()
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        todo!()
    }
}

struct EncodedBatch {
    bbox: BoundingBox,
    page_number: u32,
}

impl EncodedBatch {
    fn batches_into_batch_stream(
        batches: Vec<EncodedBatch>,
        batch_size: u32,
    ) -> SendableRecordBatchStream {
        let batches = batches
            .chunks(batch_size as usize)
            .map(|chunk| {
                let mut min_x = Vec::with_capacity(chunk.len());
                let mut min_y = Vec::with_capacity(chunk.len());
                let mut max_x = Vec::with_capacity(chunk.len());
                let mut max_y = Vec::with_capacity(chunk.len());
                let mut page_numbers = Vec::with_capacity(chunk.len());

                for item in chunk {
                    min_x.push(item.bbox.min_x);
                    min_y.push(item.bbox.min_y);
                    max_x.push(item.bbox.max_x);
                    max_y.push(item.bbox.max_y);
                    page_numbers.push(item.page_number);
                }

                RecordBatch::try_new(
                    RTREE_PAGE_SCHEMA.clone(),
                    vec![
                        Arc::new(Float64Array::from(min_x)),
                        Arc::new(Float64Array::from(min_y)),
                        Arc::new(Float64Array::from(max_x)),
                        Arc::new(Float64Array::from(max_y)),
                        Arc::new(new_null_array(&DataType::UInt64, chunk.len())),
                        Arc::new(UInt32Array::from(page_numbers)),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        Box::pin(RecordBatchStreamAdapter::new(
            RTREE_PAGE_SCHEMA.clone(),
            stream::iter(batches).map(Ok).boxed(),
        ))
    }
}

async fn train_rtree_page(
    batch: RecordBatch,
    batch_idx: u32,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedBatch> {
    // Leaf pages lack pageid, branch pages lacks rowid, fill the missing column with null.
    let columns = RTREE_PAGE_SCHEMA
        .fields()
        .iter()
        .map(|f| {
            batch
                .column_by_name(f.name())
                .map(|arr| arr.clone())
                .unwrap_or_else(|| Arc::new(new_null_array(f.data_type(), batch.num_rows())))
        })
        .collect::<Vec<_>>();
    let new_batch = RecordBatch::try_new(RTREE_PAGE_SCHEMA.clone(), columns)?;

    let bbox = analyze_batch(&new_batch)?;
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
        _ => {
            return Err(Error::Arrow {
                message: "BBOX_SCHEMA must be Float64 columns (min_x,min_y,max_x,max_y)".into(),
                location: location!(),
            })
        }
    };
    let len = min_x.len();
    for i in 0..len {
        bbox.update(
            min_x.value(i),
            min_y.value(i),
            max_x.value(i),
            max_y.value(i),
        );
    }
    Ok(bbox)
}
