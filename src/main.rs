use arrow::array::Array;
use arrow::util::pretty::pretty_format_batches;
use clap::Parser;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
use datafusion::prelude::NdJsonReadOptions;
use qlm::{CreateVectorIndexTableFunc, EmbeddingClient, LanceManager, LlmClient, LlmFoldUdaf, LlmUdf, LlmUnfoldUdf, VectorContext, VectorSearchTableFunc};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::{Hint, Hinter};
use rustyline::history::{DefaultHistory, History, SearchDirection};
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Context, EditMode, Editor, Helper};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "qlm")]
#[command(about = "SQL shell with LLM-powered UDFs")]
#[command(version)]
struct Args {
    /// SQL file to execute (if not provided, starts interactive mode)
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// Execute SQL statement and exit
    #[arg(short = 'c', long)]
    command: Option<String>,

    /// Data files to load as tables (CSV, Parquet, JSON)
    /// Use name=path syntax to specify table name, or just path to use filename
    #[arg(short = 't', long = "table", value_name = "NAME=PATH")]
    tables: Vec<String>,

    /// API key for Doubleword
    #[arg(long, env = "DOUBLEWORD_API_KEY")]
    api_key: String,

    /// API base URL
    #[arg(
        long,
        env = "DOUBLEWORD_API_URL",
        default_value = "https://api.doubleword.ai/v1"
    )]
    api_url: String,

    /// Model to use
    #[arg(
        long,
        env = "DOUBLEWORD_MODEL",
        default_value = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
    )]
    model: String,

    /// Embedding model to use
    #[arg(
        long,
        env = "DOUBLEWORD_EMBEDDING_MODEL",
        default_value = "Qwen/Qwen3-Embedding-8B"
    )]
    embedding_model: String,

    /// Embedding dimensions
    #[arg(long, env = "DOUBLEWORD_EMBEDDING_DIM", default_value = "4096")]
    embedding_dim: usize,

    /// Vector database path
    #[arg(long, env = "QLM_VECTOR_DB")]
    vector_db: Option<PathBuf>,
}

// SQL keywords for completion
const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "AS",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
    "GROUP",
    "HAVING",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "FULL",
    "CROSS",
    "ON",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "ADD",
    "COLUMN",
    "INDEX",
    "VIEW",
    "DISTINCT",
    "ALL",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "CAST",
    "COALESCE",
    "NULLIF",
    "EXISTS",
    "BETWEEN",
    "LIKE",
    "ILIKE",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "FIRST",
    "LAST",
    "TRUE",
    "FALSE",
    "WITH",
    "RECURSIVE",
    "OVER",
    "PARTITION",
    "WINDOW",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "LAG",
    "LEAD",
    "SHOW",
    "TABLES",
    "DESCRIBE",
    "EXPLAIN",
    "ANALYZE",
    // Our UDFs
    "llm",
    "llm_unfold",
    "llm_fold",
    "vector_search",
    "create_vector_index",
];

// Dot commands
const DOT_COMMANDS: &[&str] = &[
    ".help",
    ".quit",
    ".exit",
    ".tables",
    ".schema",
    ".load",
    ".functions",
    ".clear",
    ".history",
    ".index",
    ".indices",
];

struct SqlHelper {
    tables: RefCell<HashSet<String>>,
    columns: RefCell<HashSet<String>>,
}

impl SqlHelper {
    fn new() -> Self {
        Self {
            tables: RefCell::new(HashSet::new()),
            columns: RefCell::new(HashSet::new()),
        }
    }

    fn update_tables(&self, tables: HashSet<String>) {
        *self.tables.borrow_mut() = tables;
    }

    fn update_columns(&self, columns: HashSet<String>) {
        *self.columns.borrow_mut() = columns;
    }
}

impl Helper for SqlHelper {}

impl Completer for SqlHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let line_to_pos = &line[..pos];
        let trimmed = line_to_pos.trim_start();
        let mut completions = Vec::new();

        // Dot commands - handle specially
        if trimmed.starts_with('.') {
            let word_start = line_to_pos.len() - trimmed.len();
            for cmd in DOT_COMMANDS {
                if cmd.starts_with(trimmed) {
                    completions.push(Pair {
                        display: cmd.to_string(),
                        replacement: cmd.to_string(),
                    });
                }
            }
            completions.sort_by(|a, b| a.display.cmp(&b.display));
            return Ok((word_start, completions));
        }

        // SQL - find word start
        let word_start = line_to_pos
            .rfind(|c: char| c.is_whitespace() || c == '(' || c == ',')
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line_to_pos[word_start..];
        let word_upper = word.to_uppercase();
        let word_lower = word.to_lowercase();

        // Only complete if we have at least 1 char
        if !word.is_empty() {
            // SQL keywords (match case)
            for kw in SQL_KEYWORDS {
                if kw.starts_with(&word_upper) {
                    let replacement = if word
                        .chars()
                        .next()
                        .map(|c| c.is_lowercase())
                        .unwrap_or(false)
                    {
                        kw.to_lowercase()
                    } else {
                        kw.to_string()
                    };
                    completions.push(Pair {
                        display: kw.to_string(),
                        replacement,
                    });
                }
            }

            // Table names
            for table in self.tables.borrow().iter() {
                if table.to_lowercase().starts_with(&word_lower) {
                    completions.push(Pair {
                        display: table.clone(),
                        replacement: table.clone(),
                    });
                }
            }

            // Column names
            for col in self.columns.borrow().iter() {
                if col.to_lowercase().starts_with(&word_lower) {
                    completions.push(Pair {
                        display: col.clone(),
                        replacement: col.clone(),
                    });
                }
            }
        }

        // Sort and dedupe
        completions.sort_by(|a, b| a.display.cmp(&b.display));
        completions.dedup_by(|a, b| a.display == b.display);

        Ok((word_start, completions))
    }
}

// History hint - shows most recent matching command in gray
struct HistoryHint(String);

impl Hint for HistoryHint {
    fn display(&self) -> &str {
        &self.0
    }

    fn completion(&self) -> Option<&str> {
        Some(&self.0)
    }
}

impl Hinter for SqlHelper {
    type Hint = HistoryHint;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<Self::Hint> {
        if line.is_empty() || pos < line.len() {
            return None;
        }

        // Search history for matching prefix
        let history = ctx.history();
        if history.is_empty() {
            return None;
        }

        // Search backwards through history
        for i in (0..history.len()).rev() {
            if let Some(entry) = history.get(i, SearchDirection::Forward).ok().flatten() {
                let entry_str: &str = &entry.entry;
                if entry_str.starts_with(line) && entry_str != line {
                    // Return the suffix as hint
                    return Some(HistoryHint(entry_str[line.len()..].to_string()));
                }
            }
        }

        None
    }
}

impl Highlighter for SqlHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        // Cyan bold prompt
        Cow::Owned(format!("\x1b[1;36m{}\x1b[0m", prompt))
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        // Dim gray hint
        Cow::Owned(format!("\x1b[2;37m{}\x1b[0m", hint))
    }

    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // Basic SQL keyword highlighting
        let mut result = line.to_string();

        // Highlight SQL keywords (simple approach)
        for kw in SQL_KEYWORDS {
            let patterns = [
                format!(" {} ", kw),
                format!(" {}\n", kw),
                format!("({}", kw),
            ];
            for pat in &patterns {
                let colored = pat.replace(kw, &format!("\x1b[1;34m{}\x1b[0m", kw));
                result = result.replace(pat, &colored);
            }
            // Start of line
            if result.to_uppercase().starts_with(kw) {
                let len = kw.len();
                if result.len() == len
                    || !result
                        .chars()
                        .nth(len)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false)
                {
                    result = format!("\x1b[1;34m{}\x1b[0m{}", &result[..len], &result[len..]);
                }
            }
        }

        Cow::Owned(result)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        true
    }
}

impl Validator for SqlHelper {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Create DataFusion session with information_schema enabled
    let config = SessionConfig::new().with_information_schema(true);
    let ctx = SessionContext::new_with_config(config);

    // Register LLM UDFs
    let client = LlmClient::new(&args.api_url, &args.api_key, &args.model);

    // llm(template, args...) - per-row transformation
    let llm_udf = ScalarUDF::from(LlmUdf::new(client.clone()));
    ctx.register_udf(llm_udf);

    // llm_unfold(template, column, delimiter) - fan-out, returns array
    let llm_unfold = ScalarUDF::from(LlmUnfoldUdf::new(client.clone()));
    ctx.register_udf(llm_unfold);

    // llm_fold(column, reduce_prompt[, map_prompt]) - tree-reduce aggregate
    let llm_fold = AggregateUDF::from(LlmFoldUdaf::new(client));
    ctx.register_udaf(llm_fold);

    // Set up embedding client and LanceDB
    let embedding_client = EmbeddingClient::new(
        &args.api_url,
        &args.api_key,
        &args.embedding_model,
        args.embedding_dim,
    );

    let vector_db_path = args.vector_db.unwrap_or_else(|| {
        dirs::data_local_dir()
            .map(|p| p.join("qlm").join("vectors"))
            .unwrap_or_else(|| PathBuf::from(".qlm/vectors"))
    });

    // Create vector DB directory if needed
    std::fs::create_dir_all(&vector_db_path).ok();

    let lance_manager = Arc::new(LanceManager::new(vector_db_path, embedding_client));
    let vector_ctx = Arc::new(VectorContext::new(lance_manager.clone()));

    // vector_search(table, text_col, id_col, query, limit) - semantic search table function
    ctx.register_udtf("vector_search", Arc::new(VectorSearchTableFunc::new(vector_ctx.clone())));

    // create_vector_index(name, table, text_col, id_col) - explicit index creation
    ctx.register_udtf("create_vector_index", Arc::new(CreateVectorIndexTableFunc::new(vector_ctx.clone())));

    // Load any specified tables
    for table_spec in &args.tables {
        load_table(&ctx, table_spec).await?;
    }

    // Wrap context in Arc for sharing
    let ctx = Arc::new(ctx);

    // Set session on vector context for lazy index creation
    vector_ctx.set_session(ctx.clone()).await;

    // Execute based on mode
    if let Some(sql) = args.command {
        execute_sql(&ctx, &sql).await?;
    } else if let Some(file) = args.file {
        let sql = std::fs::read_to_string(&file)?;
        for statement in sql.split(';') {
            let statement = statement.trim();
            if !statement.is_empty() {
                execute_sql(&ctx, statement).await?;
            }
        }
    } else {
        run_repl(ctx, lance_manager).await?;
    }

    Ok(())
}

async fn load_table(ctx: &SessionContext, spec: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (name, path) = if let Some((name, path)) = spec.split_once('=') {
        (name.to_string(), PathBuf::from(path))
    } else {
        let path = PathBuf::from(spec);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("Invalid file path")?
            .to_string();
        (name, path)
    };

    let path_str = path.to_string_lossy();
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "csv" => {
            ctx.register_csv(&name, &path_str, Default::default())
                .await?;
            eprintln!("Loaded CSV '{}' as table '{}'", path_str, name);
        }
        "parquet" => {
            ctx.register_parquet(&name, &path_str, Default::default())
                .await?;
            eprintln!("Loaded Parquet '{}' as table '{}'", path_str, name);
        }
        ext @ ("json" | "jsonl" | "ndjson") => {
            let file_ext = format!(".{}", ext);
            let mut options = NdJsonReadOptions::default();
            options.schema_infer_max_records = 1000;
            options.file_extension = file_ext.leak();
            ctx.register_json(&name, &path_str, options).await?;
            eprintln!("Loaded JSON '{}' as table '{}'", path_str, name);
        }
        _ => {
            return Err(format!("Unsupported file format: {}", extension).into());
        }
    }

    Ok(())
}

async fn execute_sql(ctx: &SessionContext, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
    let df = ctx.sql(sql).await?;
    let batches = df.collect().await?;

    if batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
        println!("OK");
    } else {
        let formatted = pretty_format_batches(&batches)?;
        println!("{}", formatted);
    }

    Ok(())
}

fn get_table_names(ctx: &SessionContext) -> HashSet<String> {
    let mut tables = HashSet::new();
    if let Some(catalog) = ctx.catalog("datafusion") {
        for schema_name in catalog.schema_names() {
            if let Some(schema) = catalog.schema(&schema_name) {
                for table_name in schema.table_names() {
                    tables.insert(table_name);
                }
            }
        }
    }
    tables
}

async fn get_column_names(ctx: &SessionContext) -> HashSet<String> {
    let mut columns = HashSet::new();
    if let Ok(df) = ctx
        .sql("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public'")
        .await
    {
        if let Ok(batches) = df.collect().await {
            for batch in batches {
                if let Some(arr) = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                {
                    for i in 0..arr.len() {
                        if let Some(col) = arr.value(i).into() {
                            columns.insert(col.to_string());
                        }
                    }
                }
            }
        }
    }
    columns
}

async fn run_repl(ctx: Arc<SessionContext>, lance_manager: Arc<LanceManager>) -> Result<(), Box<dyn std::error::Error>> {
    print_welcome();

    // Configure rustyline
    let config = Config::builder()
        .history_ignore_space(true)
        .completion_type(CompletionType::Circular)
        .completion_prompt_limit(50)
        .edit_mode(EditMode::Emacs)
        .auto_add_history(false)
        .tab_stop(4)
        .build();

    let helper = SqlHelper::new();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(helper));

    // Load history
    let history_path = dirs::data_local_dir()
        .map(|p| p.join("qlm_history"))
        .unwrap_or_else(|| PathBuf::from(".qlm_history"));
    let _ = rl.load_history(&history_path);

    let mut buffer = String::new();

    loop {
        // Update completions with current tables and columns
        if let Some(helper) = rl.helper_mut() {
            helper.update_tables(get_table_names(&ctx));
            helper.update_columns(get_column_names(&ctx).await);
        }

        let prompt = if buffer.is_empty() {
            "qlm> "
        } else {
            "   ...> "
        };

        match rl.readline(prompt) {
            Ok(line) => {
                let line = line.trim();

                // Handle dot commands
                if buffer.is_empty() && line.starts_with('.') {
                    rl.add_history_entry(line)?;
                    match handle_dot_command(&ctx, line, &mut rl, &lance_manager).await {
                        Ok(true) => continue,
                        Ok(false) => break,
                        Err(e) => {
                            eprintln!("\x1b[31mError: {}\x1b[0m", e);
                            continue;
                        }
                    }
                }

                buffer.push_str(line);
                buffer.push(' ');

                // Check if statement is complete
                if line.ends_with(';') {
                    let sql = buffer.trim().trim_end_matches(';');
                    if !sql.is_empty() {
                        rl.add_history_entry(buffer.trim())?;
                        if let Err(e) = execute_sql(&ctx, sql).await {
                            eprintln!("\x1b[31mError: {}\x1b[0m", e);
                        }
                    }
                    buffer.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                buffer.clear();
                println!("^C (use .quit or Ctrl-D to exit)");
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(err) => {
                eprintln!("\x1b[31mError: {:?}\x1b[0m", err);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_path);

    Ok(())
}

fn print_welcome() {
    println!("\x1b[1;36m");
    println!("  ╭─────────────────────────────────────────╮");
    println!(
        "  │             qlm v{}                 │",
        env!("CARGO_PKG_VERSION")
    );
    println!("  │   SQL shell with LLM-powered UDFs      │");
    println!("  ╰─────────────────────────────────────────╯\x1b[0m");
    println!();
    println!("\x1b[1mKeyboard shortcuts:\x1b[0m");
    println!("  Tab          Autocomplete SQL keywords, tables, columns");
    println!("  Ctrl-R       Search command history");
    println!("  Ctrl-P/N     Previous/next history entry");
    println!("  Ctrl-A/E     Jump to start/end of line");
    println!("  Ctrl-W       Delete word backwards");
    println!("  Ctrl-U       Clear line");
    println!("  Ctrl-D       Exit (or .quit)");
    println!();
    println!("\x1b[1mCommands:\x1b[0m .help .tables .schema <t> .load <file> .index .indices .functions");
    println!();
    println!("\x1b[1mLLM Functions:\x1b[0m");
    println!("  llm(template, args...)              Per-row transform (1→1)");
    println!("  llm_unfold(template, col, delim)    Fan-out, returns array (1→N)");
    println!("  llm_fold(col, reduce[, map])        Tree-reduce aggregate (M→1)");
    println!();
    println!("\x1b[1mVector Functions:\x1b[0m");
    println!("  vector_search(t, col, id, query, n)     Semantic search");
    println!("  create_vector_index(name, t, col, id)   Create index explicitly");
    println!();
    println!("  Tip: vector_search creates indices lazily, {{0:9\\n}} for batching");
    println!();
}

async fn handle_dot_command(
    ctx: &SessionContext,
    cmd: &str,
    rl: &mut Editor<SqlHelper, DefaultHistory>,
    lance_manager: &Arc<LanceManager>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let command = parts.first().map(|s| s.to_lowercase()).unwrap_or_default();

    match command.as_str() {
        ".quit" | ".exit" | ".q" => Ok(false),
        ".help" | ".h" => {
            println!("\x1b[1mDot Commands:\x1b[0m");
            println!("  .help             Show this help message");
            println!("  .quit / .exit     Exit the shell");
            println!("  .tables           List all loaded tables");
            println!("  .schema <table>   Show table schema (columns and types)");
            println!("  .load <path>      Load file as table (CSV, Parquet, JSON)");
            println!("  .load name=<path> Load file with custom table name");
            println!("  .functions        List all available functions");
            println!("  .index <t> <c> <id> Create vector index from table");
            println!("  .indices          List vector indices");
            println!("  .history          Show command history");
            println!("  .clear            Clear the screen");
            println!();
            println!("\x1b[1mSQL:\x1b[0m");
            println!("  End statements with semicolon (;)");
            println!("  Multi-line statements supported");
            println!();
            println!("\x1b[1mLLM Functions:\x1b[0m");
            println!();
            println!("  llm(template, arg1, arg2, ...)");
            println!("    Per-row transform. Template uses {{0}}, {{1}}, etc.");
            println!("    1 row → 1 LLM call → 1 output");
            println!();
            println!("  llm_unfold(template, column, delimiter)");
            println!("    Fan-out. Returns array, use UNNEST to expand into rows.");
            println!("    1 row → 1 LLM call → N outputs (split by delimiter)");
            println!("    With {{0:K}}: K rows → 1 LLM call → K arrays");
            println!();
            println!("  llm_fold(column, reduce_prompt[, map_prompt])");
            println!("    Aggregate via K-way tree-reduce. K from placeholders.");
            println!("    M rows → log_K(M) LLM calls → 1 output");
            println!();
            println!("\x1b[1mRange Syntax (batching):\x1b[0m");
            println!("  {{0}}, {{1}}, ...     Single values");
            println!("  {{0:9, }}             10 values joined with ', '");
            println!("  {{0:9\\n}}             10 values joined with newline");
            println!();
            println!("\x1b[1mExamples:\x1b[0m");
            println!();
            println!("  -- Transform each row");
            println!("  SELECT llm('Translate: {{0}}', text) FROM docs;");
            println!();
            println!("  -- Fan-out: extract multiple values per row");
            println!("  SELECT d.id, item");
            println!("  FROM docs d, UNNEST(llm_unfold('List names:\\n{{0}}', d.text, '\\n')) AS t(item);");
            println!();
            println!("  -- Fan-out with batching (10 rows per LLM call)");
            println!("  SELECT d.id, item");
            println!("  FROM docs d, UNNEST(llm_unfold('Classify each:\\n{{0:9\\n}}', d.text, '\\n')) AS t(item);");
            println!();
            println!("  -- Fold: reduce all rows to one");
            println!("  SELECT llm_fold(text, 'Summarize:\\n{{0:3\\n---\\n}}') FROM docs;");
            Ok(true)
        }
        ".tables" | ".t" => {
            let tables = get_table_names(ctx);
            let user_tables: Vec<_> = tables
                .iter()
                .filter(|t| {
                    !["tables", "views", "columns", "df_settings", "schemata"].contains(&t.as_str())
                })
                .collect();

            if user_tables.is_empty() {
                println!("No tables loaded. Use .load <file> to load a table.");
            } else {
                println!("\x1b[1mTables:\x1b[0m");
                for table in user_tables {
                    println!("  {}", table);
                }
            }
            Ok(true)
        }
        ".schema" | ".s" => {
            if parts.len() < 2 {
                println!("Usage: .schema <table_name>");
            } else {
                let table = parts[1];
                let df = ctx.sql(&format!("DESCRIBE {}", table)).await?;
                let batches = df.collect().await?;
                let formatted = pretty_format_batches(&batches)?;
                println!("{}", formatted);
            }
            Ok(true)
        }
        ".load" | ".l" => {
            if parts.len() < 2 {
                println!("Usage: .load <path> or .load <name>=<path>");
                println!("Supported formats: CSV, Parquet, JSON, JSONL, NDJSON");
            } else {
                let spec = parts[1..].join(" ");
                load_table(ctx, &spec).await?;
            }
            Ok(true)
        }
        ".functions" | ".f" => {
            // Just show our custom functions prominently
            println!("\x1b[1mLLM Functions:\x1b[0m");
            println!("  llm(template, args...)              Transform (1→1)");
            println!("  llm_unfold(template, col, delim)    Fan-out (1→N)");
            println!("  llm_fold(col, reduce[, map])        Aggregate (M→1)");
            println!();
            println!("\x1b[1mVector Functions:\x1b[0m");
            println!("  vector_search(t, col, id, query, n)     Semantic search");
            println!("  create_vector_index(name, t, col, id)   Create index");
            println!();
            println!(
                "Run 'SHOW FUNCTIONS;' to see all {} available functions.",
                {
                    let df = ctx.sql("SHOW FUNCTIONS").await?;
                    let batches = df.collect().await?;
                    batches.iter().map(|b| b.num_rows()).sum::<usize>()
                }
            );
            Ok(true)
        }
        ".history" => {
            let history = rl.history();
            let len = history.len();
            let start = if len > 20 { len - 20 } else { 0 };
            println!("\x1b[1mRecent commands:\x1b[0m");
            for i in start..len {
                if let Ok(Some(entry)) = history.get(i, SearchDirection::Forward) {
                    let entry_str: &str = &entry.entry;
                    println!("  \x1b[2m{:4}\x1b[0m  {}", i + 1, entry_str);
                }
            }
            if len > 20 {
                println!("  ... ({} more)", len - 20);
            }
            Ok(true)
        }
        ".clear" | ".cls" => {
            print!("\x1b[2J\x1b[1;1H");
            Ok(true)
        }
        ".index" | ".idx" => {
            // .index <table> <text_column> <id_column>
            if parts.len() < 4 {
                println!("Usage: .index <table> <text_column> <id_column>");
                println!("Creates a vector index from a table's text column.");
                println!();
                println!("Example: .index documents content id");
            } else {
                let table_name = parts[1];
                let text_column = parts[2];
                let id_column = parts[3];
                let index_name = format!("{}_{}", table_name, text_column);

                // Query the table
                let df = ctx.sql(&format!("SELECT * FROM {}", table_name)).await?;
                let batches = df.collect().await?;

                if batches.is_empty() {
                    println!("Table '{}' is empty.", table_name);
                } else {
                    let count = lance_manager
                        .create_index(&index_name, batches, text_column, id_column)
                        .await
                        .map_err(|e| e.to_string())?;
                    println!(
                        "Created vector index '{}' with {} embeddings.",
                        index_name, count
                    );
                }
            }
            Ok(true)
        }
        ".indices" | ".vectors" => {
            let indices = lance_manager.list_indices().await.map_err(|e| e.to_string())?;
            if indices.is_empty() {
                println!("No vector indices. Use .index to create one.");
            } else {
                println!("\x1b[1mVector Indices:\x1b[0m");
                for idx in indices {
                    println!("  {}", idx);
                }
            }
            Ok(true)
        }
        _ => {
            println!(
                "Unknown command: {}. Type .help for available commands.",
                command
            );
            Ok(true)
        }
    }
}
