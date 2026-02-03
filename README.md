# qlm

SQL shell with LLM-powered user-defined functions. Query your data with `SELECT`, transform it with LLMs.

```sql
SELECT title, llm('Classify as theoretical or applied: {0}', title) as type
FROM papers
LIMIT 100;
```

qlm extends SQL with three LLM functions that let you transform, fan-out, and aggregate data using language models—all through familiar SQL syntax. Built on [DataFusion](https://datafusion.apache.org/) and the [Doubleword Batch API](https://doubleword.ai).

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

## LLM Functions

### `llm(template, args...)` — Per-row transform (1→1)

Transform each row independently. The template uses `{0}`, `{1}`, etc. for column values.

```sql
-- Classify each paper
SELECT title, llm('Is this paper about machine learning? Answer yes/no: {0}', title) as is_ml
FROM papers;

-- Translate with multiple columns
SELECT llm('Translate "{0}" from {1} to English', title, language) as translated
FROM articles;
```

### `llm_unfold(template, column, delimiter)` — Fan-out (1→N)

Extract multiple values from each row. Returns an array; use `UNNEST` to expand into rows.

```sql
-- Extract keywords from each abstract
SELECT p.title, keyword
FROM papers p, UNNEST(llm_unfold(
  'List 5 keywords for this paper, one per line:\n{0}',
  p.abstract,
  '\n'
)) AS t(keyword);

-- Parse structured data
SELECT id, entity
FROM documents d, UNNEST(llm_unfold(
  'Extract all company names mentioned:\n{0}\nOne per line.',
  d.content,
  '\n'
)) AS t(entity);
```

### `llm_fold(column, reduce_prompt[, map_prompt])` — Aggregate (M→1)

Reduce many rows to a single output via tree-reduce. Efficiently handles large datasets by combining values in a tree structure.

```sql
-- Summarize all papers in a category
SELECT primary_subject,
       llm_fold(abstract, 'Synthesize these abstracts into key themes:\n{0}\n---\n{1}') as themes
FROM papers
GROUP BY primary_subject;

-- With optional map step (transform before reducing)
SELECT llm_fold(
  content,
  'Combine these summaries:\n{0}\n---\n{1}',
  'Summarize in 2 sentences: {0}'
) as final_summary
FROM long_documents;
```

The reduce template must have at least `{0}` and `{1}`. You can use higher-arity reduces like `{0}\n{1}\n{2}\n{3}` for 4-way folding.

## Batching with Range Syntax

For efficiency, you can batch multiple rows into a single LLM call using range syntax `{start:end<separator>}`:

```sql
-- Process 10 rows per LLM call
SELECT llm('Classify each title:\n{0:9\n}', title) as classifications
FROM papers;

-- The range {0:9\n} expands to {0}\n{1}\n{2}\n...\n{9}
-- joining 10 consecutive values with newlines
```

This reduces API calls by 10x while maintaining the same output structure.

## Supported File Formats

Load data files with `-t` or `.load`:

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
qlm> .tables          # List loaded tables
qlm> .schema papers   # Show table schema
qlm> .functions       # List available functions
qlm> .help            # Show all commands
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
| `--model` | `DOUBLEWORD_MODEL` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Model to use |

## Examples

**Classify 1000 papers:**
```sql
SELECT arxiv_id, title,
       llm('Classify this paper into one category: ML, Physics, Math, CS, Other\nTitle: {0}', title) as category
FROM papers
LIMIT 1000;
```

**Extract entities and join:**
```sql
WITH entities AS (
  SELECT d.id, entity
  FROM documents d, UNNEST(llm_unfold('Extract person names:\n{0}', d.content, '\n')) AS t(entity)
)
SELECT entity, COUNT(*) as mentions
FROM entities
GROUP BY entity
ORDER BY mentions DESC;
```

**Summarize by group:**
```sql
SELECT department,
       llm_fold(report, 'Combine into executive summary:\n{0}\n---\n{1}') as summary
FROM quarterly_reports
GROUP BY department;
```

**Batch classification with range syntax:**
```sql
SELECT llm('For each title, output "theoretical" or "applied" on its own line:\n{0:19\n}', title)
FROM papers
LIMIT 1000;
```

## Best Practices

### Use subqueries to limit input to aggregates

When using `llm_fold`, place your `LIMIT` in a subquery—otherwise it limits the output (always 1 row), not the input:

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

Test your prompts on a small sample before running on the full dataset:

```sql
-- Test with 5 rows first
SELECT llm('Classify: {0}', abstract) FROM papers LIMIT 5;

-- Then scale up
SELECT llm('Classify: {0}', abstract) FROM papers LIMIT 1000;
```

### Batch inputs to fill context

Use `llm_unfold` with range syntax to pack multiple rows into a single LLM call. The delimiter splits the output back to individual rows:

```sql
-- 1 row per call: 1000 rows = 1000 LLM calls
SELECT title, classification
FROM papers p, UNNEST(llm_unfold('Classify: {0}', p.title, '\n')) AS t(classification)
LIMIT 1000;

-- 50 rows per call: 1000 rows = 20 LLM calls
SELECT title, classification
FROM papers p, UNNEST(llm_unfold(
  'Classify each paper (one word per line):\n{0:49\n}',
  p.title,
  '\n'
)) AS t(classification)
LIMIT 1000;
```

The LLM should produce one output per input, separated by the delimiter. Be explicit in your prompt:

```sql
llm_unfold('For each title, output one category per line:
{0:19\n}

Categories (one per line):', title, '\n')
```

If the LLM produces fewer outputs than inputs, missing rows get empty strings. If it produces more, extras are dropped. Neither case raises an error—so check your results when tuning prompts.

How many to batch depends on your data:
- **Short text** (titles, names): batch 20-50 per call
- **Medium text** (abstracts, paragraphs): batch 5-10 per call
- **Long text** (full documents): 1 per call, or use map-reduce

### Be specific about output format

Clear instructions about the expected output format improve consistency and make parsing easier:

```sql
-- Vague (inconsistent outputs)
SELECT llm('What kind of research is this? {0}', title) FROM papers;

-- Specific (predictable outputs)
SELECT llm('Classify as "theoretical" or "applied" (one word only): {0}', title) FROM papers;
```

For structured extraction, specify the exact format:

```sql
SELECT llm('Extract the method name and dataset used.
Format: METHOD: <name>, DATASET: <name>
If not mentioned, use "N/A".

Abstract: {0}', abstract) FROM papers;
```

### Use map-reduce for large text

When folding long documents, use the optional map prompt to summarize first:

```sql
SELECT llm_fold(
  content,
  'Combine these summaries:\n{0}\n---\n{1}',      -- reduce
  'Key points in 2 sentences: {0}'                 -- map (runs first)
) FROM large_documents;
```

### Filter before transforming

Apply WHERE clauses to reduce the dataset before LLM processing:

```sql
-- Process only relevant rows
SELECT llm('Summarize: {0}', abstract)
FROM papers
WHERE primary_subject = 'Machine Learning (cs.LG)'
LIMIT 100;
```

## How It Works

qlm uses the [Doubleword Batch API](https://docs.doubleword.ai) to process LLM requests efficiently:

1. **Batching**: Multiple prompts are combined into batch requests, reducing overhead
2. **Streaming downloads**: Results stream back as they complete, showing progress
3. **Tree-reduce**: `llm_fold` uses K-way tree reduction, requiring only O(log N) LLM calls for N rows

The batch API provides the same models as real-time inference at lower cost with higher throughput.

## License

MIT
