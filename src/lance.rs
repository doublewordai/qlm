use arrow::array::{Array, ArrayRef, Float32Array, RecordBatch, StringArray, UInt64Array};
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
        let mut ids: Vec<u64> = Vec::new();

        for batch in &batches {
            let text_col_idx = batch
                .schema()
                .index_of(text_column)
                .map_err(|_| format!("Column '{}' not found", text_column))?;

            let id_col_idx = batch
                .schema()
                .index_of(id_column)
                .map_err(|_| format!("Column '{}' not found", id_column))?;

            let text_array = batch
                .column(text_col_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| format!("Column '{}' is not a string", text_column))?;

            let id_array = batch.column(id_col_idx);

            for i in 0..text_array.len() {
                if !text_array.is_null(i) {
                    texts.push(text_array.value(i).to_string());
                    ids.push(extract_id(id_array, i).ok_or_else(|| {
                        format!("Could not extract ID from column '{}'", id_column)
                    })?);
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
        let id_array = UInt64Array::from(ids);
        let text_array = StringArray::from(texts);

        // Convert embeddings to Arrow FixedSizeList
        let dim = self.embedding_client.dimensions();
        let vector_array = create_vector_array(&embeddings, dim)?;

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
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
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let text_col = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(texts)) = (id_col, text_col) {
                for i in 0..batch.num_rows() {
                    search_results.push(SearchResult {
                        id: ids.value(i),
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
    pub id: u64,
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

        // Extract identifiers
        let table_name = extract_identifier(&args[0], "table")?;
        let text_col = extract_identifier(&args[1], "text_col")?;
        let id_col = extract_identifier(&args[2], "id_col")?;

        // Use same naming convention as lazy creation
        let index_name = format!("{}_{}", table_name, text_col);

        let ctx = self.ctx.clone();
        let count = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Get session to query table
                let session_guard = ctx.session.read().await;
                let session = session_guard.as_ref().ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "Session not initialized".to_string(),
                    )
                })?;

                // Query the table
                let df = session
                    .sql(&format!(
                        "SELECT {}, {} FROM {}",
                        id_col, text_col, table_name
                    ))
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
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("distance", DataType::Float32, false),
        ]));

        // Build arrays from results
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        let distances: Vec<f32> = results.iter().map(|r| r.distance).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(StringArray::from(texts)),
                Arc::new(Float32Array::from(distances)),
            ],
        )?;

        let table = MemTable::try_new(schema, vec![vec![batch]])?;
        Ok(Arc::new(table))
    }
}

// Helper functions

fn extract_id(array: &ArrayRef, idx: usize) -> Option<u64> {
    if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        Some(arr.value(idx))
    } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
        Some(arr.value(idx) as u64)
    } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        Some(arr.value(idx) as u64)
    } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Int32Array>() {
        Some(arr.value(idx) as u64)
    } else {
        None
    }
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
