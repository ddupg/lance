// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::HashMap;
use crate::frag_reuse::FragReuseIndex;
use crate::{pb, Index, IndexType};
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::rtree::sort::Sorter;
use crate::scalar::{AnyQuery, CreatedIndex, IndexStore, IndexWriter, ScalarIndex, SearchResult, UpdateCriteria};
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
use deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use serde_json::Value;
use tempfile::{tempdir, TempDir};
use crate::metrics::MetricsCollector;
use crate::scalar::btree::BTreeIndex;
use crate::scalar::ngram::NGramIndex;
use crate::vector::VectorIndex;

mod sort;

const DEFAULT_RTREE_PAGE_SIZE: u32 = 4096;
const RTREE_PAGES_NAME: &str = "page_data.lance";
const RTREE_INDEX_VERSION: u32 = 0;

static BBOX_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("min_x", DataType::Float64, false),
        ArrowField::new("min_y", DataType::Float64, false),
        ArrowField::new("max_x", DataType::Float64, false),
        ArrowField::new("max_y", DataType::Float64, false),
    ]))
});
static BBOX_ROWID_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    let mut fields = BBOX_SCHEMA.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(ROW_ID, DataType::UInt64, true)));
    Arc::new(ArrowSchema::new(fields))
});
const PAGE_ID_COLUMN: &str = "_page_id";
static RTREE_PAGE_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
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

#[derive(Debug, Clone, PartialEq)]
pub struct RTreeMetadata {
    pub(crate) page_size: u32,
    pub(crate) num_pages: u32,
}

impl RTreeMetadata {
    pub fn new(page_size: u32, num_pages: u32) -> Self {
        Self {
            page_size,
            num_pages,
        }
    }
}

impl From<RTreeMetadata> for HashMap<String, String> {
    fn from(metadata: RTreeMetadata) -> Self {
        HashMap::from_iter(vec![
            ("page_size".to_owned(), metadata.page_size.to_string()),
            ("num_pages".to_owned(), metadata.num_pages.to_string())
        ])
    }
}

impl From<&HashMap<String, String>> for RTreeMetadata {
    fn from(metadata: &HashMap<String, String>) -> Self {
        let page_size = metadata
            .get("page_size")
            .map(|bs| bs.parse().unwrap_or(DEFAULT_RTREE_PAGE_SIZE))
            .unwrap_or(DEFAULT_RTREE_PAGE_SIZE);
        let num_pages = metadata
            .get("num_pages")
            .map(|bs| bs.parse().unwrap_or(0))
            .unwrap_or(0);
        Self::new(page_size, num_pages)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct RTree {
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
                    location: location!(),
                })
            }
        }
        _ => Err(Error::Index {
            message: format!(
                "Unsupported geometry data type: {:?}. Expected struct<x: double, y: double>",
                geometry_field.data_type()
            ),
            location: location!(),
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
            location: location!(),
        })?;

    // Get x and y columns from the struct
    let x_column = struct_array
        .column_by_name("x")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'x' field".to_string(),
            location: location!(),
        })?;

    let y_column = struct_array
        .column_by_name("y")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'y' field".to_string(),
            location: location!(),
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

fn convert_bbox_stream(source: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
    // 创建一个流，将原始数据转换为边界框
    let bbox_stream = source
        .map_err(DataFusionError::into)
        .and_then(move |batch| async move {
            let schema = batch.schema();
            let geometry_field = schema.field(0);
            let geometry_array = batch.column(0);

            let bboxes = extract_bounding_boxes(geometry_field, geometry_array.as_ref())?;
            let mut min_x_values = Float64Array::builder(bboxes.len());
            let mut min_y_values = Float64Array::builder(bboxes.len());
            let mut max_x_values = Float64Array::builder(bboxes.len());
            let mut max_y_values = Float64Array::builder(bboxes.len());
            for bbox in bboxes {
                min_x_values.append_value(bbox.min_x);
                min_y_values.append_value(bbox.min_y);
                max_x_values.append_value(bbox.max_x);
                max_y_values.append_value(bbox.max_y);
            }

            RecordBatch::try_new(
                BBOX_ROWID_SCHEMA.clone(),
                vec![
                    Arc::new(min_x_values.finish()),
                    Arc::new(min_y_values.finish()),
                    Arc::new(max_x_values.finish()),
                    Arc::new(max_y_values.finish()),
                    // rowid
                    batch.column(1).clone(),
                ],
            )
            .map_err(DataFusionError::from)
        });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        BBOX_ROWID_SCHEMA.clone(),
        bbox_stream,
    )))
}

#[derive(Debug, Clone)]
pub struct RTreeIndexBuilderOptions {
    // TODO: Supports specified sorting algorithms
}

impl Default for RTreeIndexBuilderOptions {
    fn default() -> RTreeIndexBuilderOptions {
        RTreeIndexBuilderOptions {}
    }
}

pub struct RTreeIndexBuilder {
    _options: RTreeIndexBuilderOptions,
    _tmpdir: Arc<TempDir>,
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
            _options: options,
            _tmpdir: tmpdir,
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

        self.write_index(sorted_data, store, DEFAULT_RTREE_PAGE_SIZE)
            .await?;
        sorter.cleanup().await?;

        Ok(())
    }

    pub async fn write_index(
        &mut self,
        sorted_data: SendableRecordBatchStream,
        store: &dyn IndexStore,
        page_size: u32,
    ) -> Result<()> {
        let mut page_idx: u32 = 0;
        let mut writer = store
            .new_index_file(RTREE_PAGES_NAME, RTREE_PAGE_SCHEMA.clone())
            .await?;

        let mut current_level = Some(sorted_data);
        while let Some(mut data) = current_level.take() {
            let mut next_level = vec![];
            while let Some(batch) = data.next().await {
                let batch = batch?;
                let encoded_batch = train_rtree_page(batch, page_idx, writer.as_mut()).await?;
                page_idx += 1;
                next_level.push(encoded_batch);
            }

            if !next_level.is_empty() {
                current_level = Some(encoded_batches_into_batch_stream(next_level, page_size));
            }
        }
        writer.finish_with_metadata(RTreeMetadata::new(page_size, page_idx).into()).await?;

        Ok(())
    }
}


#[derive(Clone, Debug)]
pub struct RTreeIndex {

}

impl RTreeIndex {
    pub(crate) async fn load(
        _store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        todo!()
    }
}

impl DeepSizeOf for RTreeIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        todo!()
    }
}

#[async_trait]
impl Index for RTreeIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::NotSupported {
            source: "RTreeIndex is not vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<Value> {
        todo!()
    }

    async fn prewarm(&self) -> Result<()> {
        todo!()
    }

    fn index_type(&self) -> IndexType {
        IndexType::RTree
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        todo!()
    }
}

#[async_trait]
impl ScalarIndex for RTreeIndex {
    async fn search(&self, query: &dyn AnyQuery, metrics: &dyn MetricsCollector) -> Result<SearchResult> {
        todo!()
    }

    fn can_remap(&self) -> bool {
        todo!()
    }

    async fn remap(&self, mapping: &HashMap<u64, Option<u64>>, dest_store: &dyn IndexStore) -> Result<CreatedIndex> {
        todo!()
    }

    async fn update(&self, new_data: SendableRecordBatchStream, dest_store: &dyn IndexStore) -> Result<CreatedIndex> {
        todo!()
    }

    fn update_criteria(&self) -> UpdateCriteria {
        todo!()
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
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(RTreeIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

struct EncodedBatch {
    bbox: BoundingBox,
    page_number: u32,
}

fn encoded_batches_into_batch_stream(
    batches: Vec<EncodedBatch>,
    batch_size: u32,
) -> SendableRecordBatchStream {
    let batches = batches
        .chunks(batch_size as usize)
        .map(|chunk| {
            let mut min_x = Float64Array::builder(chunk.len());
            let mut min_y = Float64Array::builder(chunk.len());
            let mut max_x = Float64Array::builder(chunk.len());
            let mut max_y = Float64Array::builder(chunk.len());
            let mut page_numbers = UInt32Array::builder(chunk.len());

            for item in chunk {
                min_x.append_value(item.bbox.min_x);
                min_y.append_value(item.bbox.min_y);
                max_x.append_value(item.bbox.max_x);
                max_y.append_value(item.bbox.max_y);
                page_numbers.append_value(item.page_number);
            }

            RecordBatch::try_new(
                RTREE_PAGE_SCHEMA.clone(),
                vec![
                    Arc::new(min_x.finish()),
                    Arc::new(min_y.finish()),
                    Arc::new(max_x.finish()),
                    Arc::new(max_y.finish()),
                    Arc::new(new_null_array(&DataType::UInt64, chunk.len())),
                    Arc::new(page_numbers.finish()),
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
