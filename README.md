# qlm

SQL shell with LLM-powered user-defined functions. Query your data with `SELECT`, transform it with LLMs.

```sql
SELECT title, llm('Classify as theoretical or applied: {0}', title) as type
FROM papers
LIMIT 100;
```

Built on [DataFusion](https://datafusion.apache.org/), [LanceDB](https://lancedb.com/), and the [Doubleword Batch API](https://doubleword.ai).

## Installation

```bash
cargo install qlm
```

Or build from source:

```bash
git clone https://github.com/doublewordai/qlm
cd qlm
cargo build --release
```

## Quick Start

```bash
# Set your API key
export DOUBLEWORD_API_KEY="your-key-here"

# Start interactive shell with a data file
qlm -t papers.parquet

# Or run a one-off query
qlm -t papers.parquet -c "SELECT title, llm('Summarize: {0}', abstract) FROM papers LIMIT 10;"
```

Get an API key at [app.doubleword.ai](https://app.doubleword.ai).

## Transforming Rows

### Simple transformation (1 row = 1 LLM call)

Use `llm()` for straightforward per-row transformations:

```sql
-- Classify each paper
SELECT title, llm('Is this about machine learning? Answer yes/no: {0}', title) as is_ml
FROM papers;

-- Use multiple columns
SELECT llm('Translate "{0}" from {1} to English', title, language) as translated
FROM articles;
```

The template uses `{0}`, `{1}`, etc. for column values.

### Batched transformation (N rows = 1 LLM call)

For large datasets, use `llm_unfold()` with range syntax to batch multiple rows into a single LLM call:

```sql
-- Process 10 rows per LLM call (100 rows = 10 LLM calls)
SELECT title, UNNEST(llm_unfold(
  'Classify each title as theoretical or applied (one word per line):\n{0:9\n}',
  title,
  '\n'
)) as classification
FROM (SELECT title FROM papers LIMIT 100);
```

The range syntax `{0:9\n}` batches 10 consecutive values, joining them with newlines. The LLM output is split by the delimiter (`\n`) and distributed back—one result per row.

This reduces API calls by 10x with minimal prompt overhead.

**When to use each:**

| Scenario | Function | Rows per LLM call |
|----------|----------|-------------------|
| Small dataset (<100 rows) | `llm()` | 1 |
| Large dataset, short text | `llm_unfold()` with `{0:49\n}` | 50 |
| Large dataset, medium text | `llm_unfold()` with `{0:9\n}` | 10 |
| Long text (full documents) | `llm()` | 1 |

## Extracting Multiple Values

Use `llm_unfold()` to extract multiple values from each row (fan-out):

```sql
-- Extract keywords from each abstract
SELECT title, UNNEST(llm_unfold(
  'List 5 keywords for this paper, one per line:\n{0}',
  abstract,
  '\n'
)) as keyword
FROM (SELECT title, abstract FROM papers LIMIT 10);

-- Parse structured data
SELECT id, UNNEST(llm_unfold(
  'Extract all company names mentioned, one per line:\n{0}',
  content,
  '\n'
)) as company
FROM (SELECT id, content FROM documents LIMIT 10);
```

The delimiter splits the LLM output into an array. Use `UNNEST` to expand into rows.

## Aggregating Data

Use `llm_fold()` to reduce many rows to a single output via tree-reduce:

```sql
-- Summarize all papers in each category
SELECT primary_subject,
       llm_fold(abstract, 'Synthesize these abstracts into key themes:\n{0}\n---\n{1}') as themes
FROM papers
GROUP BY primary_subject;
```

The reduce template must have at least `{0}` and `{1}`. The function combines values in pairs (or larger groups) recursively until one result remains.

### K-way reduction

Use more placeholders or range syntax for higher-arity folding:

```sql
-- 4-way reduce (faster for large datasets)
SELECT llm_fold(abstract, 'Combine these four summaries:\n{0}\n---\n{1}\n---\n{2}\n---\n{3}')
FROM papers;

-- Equivalent using range syntax
SELECT llm_fold(abstract, 'Combine these summaries:\n{0:3\n---\n}')
FROM papers;
```

K-way folding reduces a dataset of N rows in O(log_K(N)) LLM calls.

### Map-reduce

Add an optional map prompt to transform values before reducing:

```sql
SELECT llm_fold(
  content,
  'Combine these summaries:\n{0}\n---\n{1}',      -- reduce prompt
  'Summarize in 2 sentences: {0}'                  -- map prompt (runs first)
) as final_summary
FROM long_documents;
```

This is useful when raw values are too long—map first to compress, then reduce.

## Semantic Search

Search your data by meaning, not just keywords:

```sql
SELECT * FROM vector_search(docs, content, id, 'how does authentication work', 5);
--                          table column  id  query                          limit
```

Returns a table with `id`, `text`, and `distance` columns. Indices are created lazily on first query.

### Basic Usage

```sql
-- Search documents (index created automatically on first query)
SELECT * FROM vector_search(docs, content, id, 'kubernetes deployment', 5);

-- Filter by similarity threshold
SELECT * FROM vector_search(docs, content, id, 'database schema', 10)
WHERE distance < 0.5;

-- Join back to source table for additional columns
SELECT d.title, d.author, v.distance
FROM vector_search(docs, content, id, 'machine learning', 5) v
JOIN docs d ON d.id = v.id
ORDER BY v.distance;
```

### Explicit Index Creation

For large tables, create the index upfront to see progress:

```sql
-- Via SQL (can use with -c flag)
SELECT * FROM create_vector_index(docs, content, id);

-- Returns: index_name (docs_content), embeddings (count)
```

**In the REPL:**
```
qlm> .index docs content id
Created vector index 'docs_content' with 1000 embeddings.
```

### Managing Indices

```
qlm> .indices              -- List all vector indices
```

Indices persist to disk (default: `~/.local/share/qlm/vectors/`) and survive restarts. Lazy-created indices use the naming convention `{table}_{column}`.

## Supported File Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Parquet | `.parquet` | Recommended for large datasets |
| CSV | `.csv` | Auto-detects schema |
| JSON | `.json`, `.jsonl`, `.ndjson` | Newline-delimited JSON |

```bash
# Load multiple files
qlm -t papers=data/papers.parquet -t authors=data/authors.csv

# Or load interactively
qlm> .load papers.parquet
qlm> .load authors=people.csv
```

## Interactive Shell

The REPL includes syntax highlighting, tab completion, and command history:

```
qlm> SELECT title FROM papers WHERE primary_subject LIKE '%Physics%' LIMIT 5;
qlm> .tables          -- List loaded tables
qlm> .schema papers   -- Show table schema
qlm> .functions       -- List available functions
qlm> .index t col id  -- Create vector index
qlm> .indices         -- List vector indices
qlm> .help            -- Show all commands
```

**Keyboard shortcuts:**
- `Tab` — Autocomplete SQL keywords, table names, columns
- `Ctrl-R` — Search command history
- `Ctrl-P/N` — Previous/next history entry
- `Ctrl-D` — Exit

## Configuration

| Flag | Env Variable | Default | Description |
|------|-------------|---------|-------------|
| `--api-key` | `DOUBLEWORD_API_KEY` | (required) | API key |
| `--api-url` | `DOUBLEWORD_API_URL` | `https://api.doubleword.ai/v1` | API endpoint |
| `--model` | `DOUBLEWORD_MODEL` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | LLM model |
| `--embedding-model` | `DOUBLEWORD_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Embedding model |
| `--embedding-dim` | `DOUBLEWORD_EMBEDDING_DIM` | `4096` | Embedding dimensions |
| `--vector-db` | `QLM_VECTOR_DB` | `~/.local/share/qlm/vectors` | Vector index storage |

## Best Practices

### Use subqueries to limit input to aggregates

When using `llm_fold`, place `LIMIT` in a subquery—otherwise it limits the output (always 1 row), not the input:

```sql
-- Wrong: LIMIT applies after the fold (no effect)
SELECT llm_fold(title, 'Summarize themes:\n{0}\n---\n{1}')
FROM papers
LIMIT 20;

-- Right: LIMIT applies to the input
SELECT llm_fold(title, 'Summarize themes:\n{0}\n---\n{1}')
FROM (SELECT title FROM papers LIMIT 20);
```

### Start small, then scale

Test prompts on a small sample before running on the full dataset:

```sql
-- Test with 5 rows first
SELECT llm('Classify: {0}', abstract) FROM papers LIMIT 5;

-- Then scale up
SELECT llm('Classify: {0}', abstract) FROM papers LIMIT 1000;
```

### Be specific about output format

Clear format instructions improve consistency:

```sql
-- Vague (inconsistent outputs)
SELECT llm('What kind of research is this? {0}', title) FROM papers;

-- Specific (predictable outputs)
SELECT llm('Classify as "theoretical" or "applied" (one word only): {0}', title) FROM papers;
```

### Filter before transforming

Apply WHERE clauses to reduce the dataset before LLM processing:

```sql
SELECT llm('Summarize: {0}', abstract)
FROM papers
WHERE primary_subject = 'Machine Learning (cs.LG)'
LIMIT 100;
```

## How It Works

qlm uses the [Doubleword Batch API](https://docs.doubleword.ai) to process LLM requests efficiently:

1. **Batching**: Prompts are submitted as batch jobs, reducing overhead
2. **Streaming**: Results stream back as they complete, showing progress
3. **Tree-reduce**: `llm_fold` uses K-way reduction, requiring only O(log N) LLM calls

The batch API provides the same models as real-time inference at lower cost.

## Function Reference

### `llm(template, args...)`

Per-row transformation. Returns one string per row.

**Arguments:**
- `template` — Prompt template with `{0}`, `{1}`, etc. placeholders
- `args...` — Column values to substitute into the template

**Example:**
```sql
SELECT llm('Translate to French: {0}', title) FROM papers;
SELECT llm('Compare {0} and {1}', title, abstract) FROM papers;
```

**Notes:**
- Does not support range syntax—use `llm_unfold` for batching
- 1 row = 1 LLM call

---

### `llm_unfold(template, column, delimiter)`

Fan-out or batched transformation. Returns an array of strings per row.

**Arguments:**
- `template` — Prompt template with `{0}` or range syntax like `{0:9\n}`
- `column` — The column to process
- `delimiter` — String to split the LLM output on

**Returns:** `List<Utf8>` — use `UNNEST` to expand into rows

**Modes:**

*Fan-out (1 row → N outputs):*
```sql
SELECT id, UNNEST(llm_unfold('List keywords:\n{0}', abstract, '\n')) as keyword
FROM (SELECT id, abstract FROM papers LIMIT 10);
```

*Batched (N rows → 1 LLM call → N outputs):*
```sql
SELECT id, UNNEST(llm_unfold('Process each:\n{0:9\n}', title, '\n')) as result
FROM (SELECT id, title FROM papers LIMIT 100);
```

**Range syntax:** `{start:end<separator>}`
- `{0:9\n}` — Batch 10 values (indices 0-9), joined with newlines
- `{0:4, }` — Batch 5 values, joined with ", "

---

### `llm_fold(column, reduce_prompt[, map_prompt])`

Aggregate via tree-reduce. Returns one string for the group.

**Arguments:**
- `column` — The column to aggregate
- `reduce_prompt` — Template with at least `{0}` and `{1}` for combining values
- `map_prompt` (optional) — Template with `{0}` to transform values before reducing

**Example:**
```sql
-- Simple 2-way reduce
SELECT llm_fold(abstract, 'Combine:\n{0}\n---\n{1}') FROM papers;

-- 4-way reduce with range syntax
SELECT llm_fold(abstract, 'Combine:\n{0:3\n---\n}') FROM papers;

-- Map-reduce
SELECT llm_fold(content, 'Combine:\n{0}\n---\n{1}', 'Summarize: {0}') FROM docs;
```

**Range syntax:** The number of placeholders determines K-way arity:
- `{0}\n{1}` — 2-way (default)
- `{0:3\n}` — 4-way (indices 0-3)

**Complexity:** O(log_K(N)) LLM calls for N rows

---

### `vector_search(table, text_col, id_col, query, limit)`

Table function for semantic search. Returns rows matching the query by meaning.

**Arguments:**
- `table` — Source table name (identifier, no quotes)
- `text_col` — Column containing text to search (identifier)
- `id_col` — Column containing row IDs for joining (identifier)
- `query` — Natural language search query (string)
- `limit` — Maximum number of results to return (integer)

**Returns:** Table with columns:
- `id` — Row ID from the original data
- `text` — The indexed text
- `distance` — Cosine distance (lower = more similar)

**Example:**
```sql
-- Basic search (index created lazily)
SELECT * FROM vector_search(docs, content, id, 'error handling', 5);

-- Filter and join
SELECT d.title, v.distance
FROM vector_search(docs, content, id, 'async programming', 10) v
JOIN docs d ON d.id = v.id
WHERE v.distance < 0.8
ORDER BY v.distance;
```

**Notes:**
- Creates index automatically on first query (named `{table}_{text_col}`)
- Uses batch API for embeddings
- Indices persist across sessions

---

### `create_vector_index(table, text_col, id_col)`

Explicitly create a vector index. Use this when you want to see embedding progress for large tables.

**Arguments:**
- `table` — Source table name (identifier, no quotes)
- `text_col` — Column containing text to embed (identifier)
- `id_col` — Column containing row IDs (identifier)

**Returns:** Table with columns:
- `index_name` — Name of the created index (`{table}_{text_col}`)
- `embeddings` — Number of embeddings created

**Example:**
```sql
-- Create index explicitly (shows progress)
SELECT * FROM create_vector_index(docs, content, id);

-- Then search
SELECT * FROM vector_search(docs, content, id, 'my query', 5);
```

**Notes:**
- Index name follows the convention `{table}_{text_col}`
- Overwrites existing index with the same name
- Shows progress during embedding generation

## License

MIT
