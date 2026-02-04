use arrow::array::{Array, ArrayRef, Float32Array, RecordBatch, StringArray, StringViewArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::TableFunctionImpl;
use datafusion::catalog::TableProvider;
use datafusion::common::{plan_err, Result as DFResult, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use half::f16;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::embedding::EmbeddingClient;

/// Manages LanceDB connections and vector indices
#[derive(Debug)]
pub struct LanceManager {
    db_path: PathBuf,
    embedding_client: EmbeddingClient,
}

impl LanceManager {
    pub fn new(db_path: PathBuf, embedding_client: EmbeddingClient) -> Self {
        Self {
            db_path,
            embedding_client,
        }
    }

    /// Create a vector index from a DataFusion RecordBatch
    pub async fn create_index(
        &self,
        name: &str,
        batches: Vec<RecordBatch>,
        text_column: &str,
        id_column: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        if batches.is_empty() {
            return Ok(0);
        }

        // Collect all text values and IDs
        let mut texts: Vec<String> = Vec::new();
        let mut ids: Vec<String> = Vec::new();

        for batch in &batches {
            let text_col_idx = batch
                .schema()
                .index_of(text_column)
                .map_err(|_| format!("Column '{}' not found", text_column))?;

            let id_col_idx = batch
                .schema()
                .index_of(id_column)
                .map_err(|_| format!("Column '{}' not found", id_column))?;

            let text_col = batch.column(text_col_idx);
            let id_col = batch.column(id_col_idx);

            // Handle both Utf8 (StringArray) and Utf8View (StringViewArray) for text
            let text_values: Vec<Option<&str>> = extract_string_values(text_col)
                .ok_or_else(|| format!("Column '{}' is not a string", text_column))?;

            // Handle string or numeric IDs
            let id_values: Vec<Option<String>> = extract_id_values(id_col)
                .ok_or_else(|| format!("Column '{}' has unsupported type for ID", id_column))?;

            for (text_opt, id_opt) in text_values.into_iter().zip(id_values.into_iter()) {
                if let (Some(text), Some(id)) = (text_opt, id_opt) {
                    texts.push(text.to_string());
                    ids.push(id);
                }
            }
        }

        if texts.is_empty() {
            return Ok(0);
        }

        let count = texts.len();

        // Generate embeddings via batch API
        let embeddings = self.embedding_client.embed_batch(texts.clone()).await?;

        // Build Arrow arrays for LanceDB
        let id_array = StringArray::from(ids);
        let text_array = StringArray::from(texts);

        // Convert embeddings to Arrow FixedSizeList
        let dim = self.embedding_client.dimensions();
        let vector_array = create_vector_array(&embeddings, dim)?;

        // Create schema (id is string to support both numeric and string IDs)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float16, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(id_array), Arc::new(text_array), vector_array],
        )?;

        // Connect to LanceDB and create table
        let db = connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await?;

        // Drop existing table if it exists
        let _ = db.drop_table(name).await;

        db.create_table(name, Box::new(batches_to_reader(vec![batch])))
            .execute()
            .await?;

        Ok(count)
    }

    /// Check if an index exists
    pub async fn index_exists(&self, name: &str) -> bool {
        if let Ok(db) = connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await
        {
            if let Ok(tables) = db.table_names().execute().await {
                return tables.contains(&name.to_string());
            }
        }
        false
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        table_name: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Get query embedding
        let embeddings = self
            .embedding_client
            .embed_batch_quiet(vec![query.to_string()])
            .await?;
        let query_vector = embeddings
            .into_iter()
            .next()
            .ok_or("Failed to get query embedding")?;

        // Convert to f16 for search
        let query_f16: Vec<f16> = query_vector.iter().map(|&f| f16::from_f32(f)).collect();

        // Connect and search
        let db = connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await?;

        let table = db.open_table(table_name).execute().await?;

        let results = table
            .vector_search(query_f16)?
            .limit(limit)
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = results.try_collect().await?;

        // Parse results
        let mut search_results = Vec::new();
        for batch in batches {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let text_col = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(texts)) = (id_col, text_col) {
                for i in 0..batch.num_rows() {
                    search_results.push(SearchResult {
                        id: ids.value(i).to_string(),
                        text: texts.value(i).to_string(),
                        distance: dist_col.map(|d| d.value(i)).unwrap_or(0.0),
                    });
                }
            }
        }

        Ok(search_results)
    }

    /// List all vector indices
    pub async fn list_indices(&self) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>
    {
        let db = connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await?;
        Ok(db.table_names().execute().await?)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub distance: f32,
}

/// Shared context for vector operations - holds manager and session reference
pub struct VectorContext {
    pub manager: Arc<LanceManager>,
    pub session: Arc<RwLock<Option<Arc<SessionContext>>>>,
}

impl VectorContext {
    pub fn new(manager: Arc<LanceManager>) -> Self {
        Self {
            manager,
            session: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn set_session(&self, ctx: Arc<SessionContext>) {
        let mut session = self.session.write().await;
        *session = Some(ctx);
    }
}

impl std::fmt::Debug for VectorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorContext")
            .field("manager", &self.manager)
            .finish()
    }
}

/// Table function for vector search: vector_search(table, text_col, id_col, query, limit)
/// Returns a table with id, text, distance columns
/// Creates index lazily if it doesn't exist
#[derive(Debug)]
pub struct VectorSearchTableFunc {
    ctx: Arc<VectorContext>,
}

impl VectorSearchTableFunc {
    pub fn new(ctx: Arc<VectorContext>) -> Self {
        Self { ctx }
    }
}

/// Table function for creating vector indices: create_vector_index(table, text_col, id_col)
/// Returns a single row with the count of embeddings created
#[derive(Debug)]
pub struct CreateVectorIndexTableFunc {
    ctx: Arc<VectorContext>,
}

impl CreateVectorIndexTableFunc {
    pub fn new(ctx: Arc<VectorContext>) -> Self {
        Self { ctx }
    }
}

impl TableFunctionImpl for CreateVectorIndexTableFunc {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        if args.len() != 3 {
            return plan_err!(
                "create_vector_index requires 3 arguments: (table, text_col, id_col)"
            );
        }

        let table_name = extract_identifier(&args[0], "table")?;
        let text_col = extract_identifier(&args[1], "text_col")?;
        let id_col = extract_identifier(&args[2], "id_col")?;
        let index_name = format!("{}_{}", table_name, text_col);

        let ctx = self.ctx.clone();
        let count = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let session_guard = ctx.session.read().await;
                let session = session_guard.as_ref().ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "Session not initialized".to_string(),
                    )
                })?;

                let df = session
                    .sql(&format!("SELECT {}, {} FROM {}", id_col, text_col, table_name))
                    .await?;
                let batches = df.collect().await?;

                eprintln!("Creating vector index '{}'...", index_name);
                let count = ctx
                    .manager
                    .create_index(&index_name, batches, &text_col, &id_col)
                    .await
                    .map_err(|e| {
                        datafusion::error::DataFusionError::Execution(format!("{}", e))
                    })?;
                eprintln!("Created index '{}' with {} embeddings.", index_name, count);

                Ok::<_, datafusion::error::DataFusionError>(count)
            })
        })?;

        // Return a table with the count
        let schema = Arc::new(Schema::new(vec![
            Field::new("index_name", DataType::Utf8, false),
            Field::new("embeddings", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![index_name.as_str()])),
                Arc::new(UInt64Array::from(vec![count as u64])),
            ],
        )?;

        let table = MemTable::try_new(schema, vec![vec![batch]])?;
        Ok(Arc::new(table))
    }
}

/// Extract identifier name from an Expr (handles Column and Literal)
fn extract_identifier(expr: &Expr, arg_name: &str) -> DFResult<String> {
    match expr {
        Expr::Column(col) => Ok(col.name.clone()),
        Expr::Literal(ScalarValue::Utf8(Some(s))) => Ok(s.clone()),
        _ => plan_err!("{} must be an identifier or string", arg_name),
    }
}

impl TableFunctionImpl for VectorSearchTableFunc {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        if args.len() != 5 {
            return plan_err!(
                "vector_search requires 5 arguments: (table, text_col, id_col, query, limit)"
            );
        }

        // Extract identifiers for table, text_col, id_col
        let table_name = extract_identifier(&args[0], "table")?;
        let text_col = extract_identifier(&args[1], "text_col")?;
        let id_col = extract_identifier(&args[2], "id_col")?;

        // Extract query string
        let query = match &args[3] {
            Expr::Literal(ScalarValue::Utf8(Some(s))) => s.clone(),
            _ => return plan_err!("query must be a string literal"),
        };

        // Extract limit
        let limit: usize = match &args[4] {
            Expr::Literal(ScalarValue::Int64(Some(n))) => *n as usize,
            Expr::Literal(ScalarValue::Int32(Some(n))) => *n as usize,
            _ => return plan_err!("limit must be an integer"),
        };

        // Index name convention: table_textcol
        let index_name = format!("{}_{}", table_name, text_col);

        let ctx = self.ctx.clone();
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Check if index exists, create if not
                if !ctx.manager.index_exists(&index_name).await {
                    // Get session to query table
                    let session_guard = ctx.session.read().await;
                    let session = session_guard.as_ref().ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "Session not initialized".to_string(),
                        )
                    })?;

                    // Query the table
                    let df = session
                        .sql(&format!("SELECT {}, {} FROM {}", id_col, text_col, table_name))
                        .await?;
                    let batches = df.collect().await?;

                    // Create the index
                    eprintln!("Creating vector index '{}'...", index_name);
                    let count = ctx
                        .manager
                        .create_index(&index_name, batches, &text_col, &id_col)
                        .await
                        .map_err(|e| {
                            datafusion::error::DataFusionError::Execution(format!("{}", e))
                        })?;
                    eprintln!("Created index '{}' with {} embeddings.", index_name, count);
                }

                // Perform search
                ctx.manager
                    .search(&index_name, &query, limit)
                    .await
                    .map_err(|e| datafusion::error::DataFusionError::Execution(format!("{}", e)))
            })
        })?;

        // Build schema: id, text, distance
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("distance", DataType::Float32, false),
        ]));

        // Build arrays from results
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        let distances: Vec<f32> = results.iter().map(|r| r.distance).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(texts)),
                Arc::new(Float32Array::from(distances)),
            ],
        )?;

        let table = MemTable::try_new(schema, vec![vec![batch]])?;
        Ok(Arc::new(table))
    }
}

// Helper functions

/// Extract string values from a column (handles Utf8 and Utf8View)
fn extract_string_values(array: &ArrayRef) -> Option<Vec<Option<&str>>> {
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect())
    } else if let Some(arr) = array.as_any().downcast_ref::<StringViewArray>() {
        Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect())
    } else {
        None
    }
}

/// Extract ID values as strings (handles string and numeric types)
fn extract_id_values(array: &ArrayRef) -> Option<Vec<Option<String>>> {
    // Try string types first
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<StringViewArray>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    // Try integer types
    if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Int32Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    // Try float types (for IDs like "2312.06865" parsed as floats)
    if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Float32Array>() {
        return Some((0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect());
    }
    None
}

fn create_vector_array(
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<ArrayRef, Box<dyn std::error::Error + Send + Sync>> {
    use arrow::array::FixedSizeListBuilder;

    let mut builder =
        FixedSizeListBuilder::new(arrow::array::Float16Builder::new(), dim as i32);

    for embedding in embeddings {
        let values = builder.values();
        for &val in embedding {
            values.append_value(f16::from_f32(val));
        }
        builder.append(true);
    }

    Ok(Arc::new(builder.finish()))
}

fn batches_to_reader(
    batches: Vec<RecordBatch>,
) -> impl arrow::record_batch::RecordBatchReader + Send + 'static {
    let schema = batches[0].schema();
    arrow::record_batch::RecordBatchIterator::new(batches.into_iter().map(Ok), schema)
}

use futures_util::TryStreamExt;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, Int64Array, UInt32Array};

    #[test]
    fn test_extract_string_values_utf8() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["hello", "world", "test"]));
        let result = extract_string_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("hello"));
        assert_eq!(result[1], Some("world"));
        assert_eq!(result[2], Some("test"));
    }

    #[test]
    fn test_extract_string_values_utf8_with_nulls() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));
        let result = extract_string_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("a"));
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some("c"));
    }

    #[test]
    fn test_extract_string_values_utf8view() {
        let arr: ArrayRef = Arc::new(StringViewArray::from(vec!["view1", "view2"]));
        let result = extract_string_values(&arr).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Some("view1"));
        assert_eq!(result[1], Some("view2"));
    }

    #[test]
    fn test_extract_string_values_invalid_type() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let result = extract_string_values(&arr);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_string_values_empty_array() {
        let arr: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
        let result = extract_string_values(&arr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_id_values_string() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["id1", "id2", "id3"]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("id1".to_string()));
        assert_eq!(result[1], Some("id2".to_string()));
        assert_eq!(result[2], Some("id3".to_string()));
    }

    #[test]
    fn test_extract_id_values_int64() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![100, 200, 300]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("100".to_string()));
        assert_eq!(result[1], Some("200".to_string()));
        assert_eq!(result[2], Some("300".to_string()));
    }

    #[test]
    fn test_extract_id_values_uint64() {
        let arr: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, 2u64, 3u64]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("1".to_string()));
        assert_eq!(result[1], Some("2".to_string()));
        assert_eq!(result[2], Some("3".to_string()));
    }

    #[test]
    fn test_extract_id_values_int32() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("10".to_string()));
        assert_eq!(result[1], Some("20".to_string()));
        assert_eq!(result[2], Some("30".to_string()));
    }

    #[test]
    fn test_extract_id_values_uint32() {
        let arr: ArrayRef = Arc::new(UInt32Array::from(vec![5u32, 10u32, 15u32]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("5".to_string()));
        assert_eq!(result[1], Some("10".to_string()));
        assert_eq!(result[2], Some("15".to_string()));
    }

    #[test]
    fn test_extract_id_values_float64() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("1.5".to_string()));
        assert_eq!(result[1], Some("2.5".to_string()));
        assert_eq!(result[2], Some("3.5".to_string()));
    }

    #[test]
    fn test_extract_id_values_float32() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![1.1f32, 2.2f32]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 2);
        // Float formatting may vary slightly
        assert!(result[0].as_ref().unwrap().starts_with("1.1"));
        assert!(result[1].as_ref().unwrap().starts_with("2.2"));
    }

    #[test]
    fn test_extract_id_values_with_nulls() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some("1".to_string()));
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some("3".to_string()));
    }

    #[test]
    fn test_extract_id_values_string_view() {
        let arr: ArrayRef = Arc::new(StringViewArray::from(vec!["sv-id-1", "sv-id-2"]));
        let result = extract_id_values(&arr).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Some("sv-id-1".to_string()));
        assert_eq!(result[1], Some("sv-id-2".to_string()));
    }

    #[test]
    fn test_create_vector_array_basic() {
        let embeddings = vec![
            vec![0.1f32, 0.2f32, 0.3f32],
            vec![0.4f32, 0.5f32, 0.6f32],
        ];
        let result = create_vector_array(&embeddings, 3).unwrap();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_create_vector_array_single() {
        let embeddings = vec![vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]];
        let result = create_vector_array(&embeddings, 4).unwrap();

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_create_vector_array_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = create_vector_array(&embeddings, 3).unwrap();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_create_vector_array_high_dimensions() {
        let dim = 4096;
        let embeddings = vec![
            (0..dim).map(|i| i as f32 / dim as f32).collect(),
            (0..dim).map(|i| (i as f32 + 0.5) / dim as f32).collect(),
        ];
        let result = create_vector_array(&embeddings, dim).unwrap();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_search_result_struct() {
        let result = SearchResult {
            id: "test-id".to_string(),
            text: "test content".to_string(),
            distance: 0.123,
        };

        assert_eq!(result.id, "test-id");
        assert_eq!(result.text, "test content");
        assert!((result.distance - 0.123).abs() < f32::EPSILON);
    }

    #[test]
    fn test_search_result_clone() {
        let result = SearchResult {
            id: "id".to_string(),
            text: "text".to_string(),
            distance: 0.5,
        };

        let cloned = result.clone();
        assert_eq!(cloned.id, result.id);
        assert_eq!(cloned.text, result.text);
        assert!((cloned.distance - result.distance).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extract_identifier_column() {
        use datafusion::common::Column;

        let expr = Expr::Column(Column::from_name("my_column"));
        let result = extract_identifier(&expr, "test").unwrap();
        assert_eq!(result, "my_column");
    }

    #[test]
    fn test_extract_identifier_literal_string() {
        let expr = Expr::Literal(ScalarValue::Utf8(Some("literal_value".to_string())));
        let result = extract_identifier(&expr, "test").unwrap();
        assert_eq!(result, "literal_value");
    }

    #[test]
    fn test_extract_identifier_invalid_type() {
        let expr = Expr::Literal(ScalarValue::Int64(Some(42)));
        let result = extract_identifier(&expr, "test_arg");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test_arg"));
    }

    // Note: Integration tests for VectorContext and LanceManager
    // would require a real EmbeddingClient, which is better done in
    // integration tests with mocked API responses.
}
