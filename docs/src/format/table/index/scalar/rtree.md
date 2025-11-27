# R-Tree Index

The R-Tree index is a static, immutable 2D spatial index. It is built on bounding boxes to organize the data. This index is intended to accelerate rectangle-based pruning.

It is designed a multi-level hierarchical structure: leaf pages store tuples `(bbox, id=rowid)` for indexed geometries; branch pages aggregate child bounding boxes and store `id=pageid` pointing to child pages; a single root page encloses the entire dataset. Conceptually, it can be thought of as an extension of the B+-tree to multidimensional objects, where bounding boxes act as keys for spatial pruning.

Sorting does not change the R-Tree data structure, but it is critical to performance. Currently, Hilbert sorting is implemented, but the design is extensible to other spatial sorting algorithms.

| Sort algorithm | Description                                                                |
|----------------|----------------------------------------------------------------------------|
| Hilbert        | Uses the Hilbert curve to impose a linear ordering on data bounding boxes. |

## Index Details

```protobuf
%%% proto.message.RTreeIndexDetails %%%
```

## Storage Layout

The R-Tree index consists of two files:

1. `page_data.lance` - Stores all pages (leaf, branch) as repeated `(bbox, id)` tuples, written bottom-up (leaves first, then branch levels)
2. `nulls.lance` - Stores a serialized RowIdTreeMap of rows with null

### Page File Schema

| Column | Type      | Nullable | Description                                                                                                                                                                                                                                                                     |
|:-------|:----------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bbox` | RectType  | false    | Type is Rect defined by [geoarrow-rs](https://github.com/geoarrow/geoarrow-rs) RectType; physical storage is Struct<xmin: {FloatType}, ymin: {FloatType}, xmax: {FloatType}, ymax: {FloatType}>. Represents the node bounding box (leaf: item bbox; branch: child aggregation). |
| `id`   | UInt64    | false    | Reuse the `id` column to store `rowid` in leaf pages and `pageid` in branch pages                                                                                                                                                                                               |

### Nulls File Schema

| Column  | Type   | Nullable | Description                                                |
|:--------|:-------|:---------|:-----------------------------------------------------------|
| `nulls` | Binary | true     | Serialized RowIdTreeMap of rows with null/invalid geometry |

### Schema Metadata

The following optional keys can be used by implementations and are stored in the schema metadata:

| Key         | Type   | Description                                       |
|:------------|:-------|:--------------------------------------------------|
| `page_size` | String | Page size per page (default: "4096")              |
| `num_pages` | String | Total number of pages written                     |
| `num_items` | String | Number of non-null leaf items in the index        |
| `bbox`      | String | JSON-serialized global BoundingBox of the dataset |

## Accelerated Queries

The R-Tree index accelerates the following query types by returning a candidate set of matching bounding boxes. Exact geometry verification must be performed by the execution engine.

| Query Type     | Description                | Operation                                     | Result Type |
|:---------------|:---------------------------|:----------------------------------------------|:------------|
| **Intersects** | `St_Intersects(col, geom)` | Prunes candidates by bbox intersection        | AtMost      |
| **Contains**   | `St_Contains(col, geom)`   | Prunes candidates by bbox containment         | AtMost      |
| **Within**     | `St_Within(col, geom)`     | Prunes candidates by bbox within relation     | AtMost      |
| **Touches**    | `St_Touches(col, geom)`    | Prunes candidates by bbox touch relation      | AtMost      |
| **Crosses**    | `St_Crosses(col, geom)`    | Prunes candidates by bbox crossing relation   | AtMost      |
| **Overlaps**   | `St_Overlaps(col, geom)`   | Prunes candidates by bbox overlap relation    | AtMost      |
| **Covers**     | `St_Covers(col, geom)`     | Prunes candidates by bbox cover relation      | AtMost      |
| **CoveredBy**  | `St_Coveredby(col, geom)`  | Prunes candidates by bbox covered-by relation | AtMost      |
| **IsNull**     | `col IS NULL`              | Returns rows recorded in the nulls file       | Exact       |
