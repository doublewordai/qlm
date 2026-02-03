use arrow::array::{
    Array, ArrayRef, GenericListBuilder, StringArray, StringBuilder, StringViewArray,
};
use arrow::datatypes::{DataType, Field};
use datafusion::common::Result as DFResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use std::any::Any;
use std::sync::Arc;

use crate::client::LlmClient;
use crate::validation::{expand_template, parse_template, validate_template};

/// Helper to extract string values from either StringArray or StringViewArray
enum StringArrayRef<'a> {
    String(&'a StringArray),
    StringView(&'a StringViewArray),
}

impl<'a> StringArrayRef<'a> {
    fn try_from_array(arr: &'a ArrayRef) -> Option<Self> {
        if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
            Some(StringArrayRef::String(s))
        } else if let Some(s) = arr.as_any().downcast_ref::<StringViewArray>() {
            Some(StringArrayRef::StringView(s))
        } else {
            None
        }
    }

    fn value(&self, i: usize) -> &str {
        match self {
            StringArrayRef::String(arr) => arr.value(i),
            StringArrayRef::StringView(arr) => arr.value(i),
        }
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            StringArrayRef::String(arr) => arr.is_null(i),
            StringArrayRef::StringView(arr) => arr.is_null(i),
        }
    }
}

/// Variadic LLM UDF: llm(template, arg1, arg2, ...)
/// Template uses {0}, {1}, {2}, etc. for placeholders
/// Example: llm('Translate {0} to {1}', text_col, 'French')
#[derive(Debug)]
pub struct LlmUdf {
    signature: Signature,
    client: LlmClient,
}

impl LlmUdf {
    pub fn new(client: LlmClient) -> Self {
        Self {
            // At least 1 argument (template), then any number of string args
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Volatile),
            client,
        }
    }
}

/// LLM Unfold UDF: llm_unfold(template, column, delimiter)
///
/// Fan-out operation: each row produces multiple output values.
/// Returns an array; use UNNEST to expand into rows.
///
/// Per-row mode (no range syntax):
///   llm_unfold('Extract names:\n{0}', content, '\n')
///   → Each row = 1 LLM call, output split into array
///
/// Batched mode (with range syntax like {0:9}):
///   llm_unfold('Classify each:\n{0:9\n}\nOne per line.', content, '\n')
///   → Every 10 rows = 1 LLM call, output split and distributed back
///
/// Use with UNNEST to expand arrays into rows:
///   SELECT d.id, item FROM docs d, UNNEST(llm_unfold(...)) as t(item)
#[derive(Debug)]
pub struct LlmUnfoldUdf {
    signature: Signature,
    client: LlmClient,
}

impl LlmUnfoldUdf {
    pub fn new(client: LlmClient) -> Self {
        Self {
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Volatile),
            client,
        }
    }
}

impl ScalarUDFImpl for LlmUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "llm"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let args = &args.args;
        if args.is_empty() {
            return Err(datafusion::error::DataFusionError::Execution(
                "llm() requires at least a template argument".to_string(),
            ));
        }

        // Get the number of rows from the first array argument
        let num_rows = args
            .iter()
            .find_map(|a| match a {
                ColumnarValue::Array(arr) => Some(arr.len()),
                _ => None,
            })
            .unwrap_or(1);

        // Convert all arguments to arrays of the same length
        let arrays: Vec<ArrayRef> = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(arr) => Ok(arr.clone()),
                ColumnarValue::Scalar(s) => s.to_array_of_size(num_rows),
            })
            .collect::<DFResult<Vec<_>>>()?;

        // Convert to string arrays (supports both Utf8 and Utf8View)
        let string_arrays: Vec<StringArrayRef> = arrays
            .iter()
            .map(|arr| {
                StringArrayRef::try_from_array(arr).ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "All arguments to llm() must be strings".to_string(),
                    )
                })
            })
            .collect::<DFResult<Vec<_>>>()?;

        // Build prompts by filling in templates
        let mut prompts = Vec::with_capacity(num_rows);
        let mut null_indices = Vec::new();

        for row in 0..num_rows {
            // Check for nulls
            if string_arrays.iter().any(|arr| arr.is_null(row)) {
                null_indices.push(row);
                prompts.push(String::new()); // Placeholder
                continue;
            }

            // Get template (first arg)
            let template = string_arrays[0].value(row);

            // Validate template on first non-null row
            if row == 0 || (null_indices.len() == row) {
                let arg_count = string_arrays.len() - 1; // Exclude template itself
                match validate_template(template, arg_count, false) {
                    Ok(warnings) => {
                        for warning in warnings {
                            eprintln!("llm() warning: {}", warning);
                        }
                    }
                    Err(e) => {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Invalid template: {}",
                            e
                        )));
                    }
                }
            }

            // Fill in placeholders {0}, {1}, {2}, etc.
            let mut prompt = template.to_string();
            for (i, arr) in string_arrays.iter().skip(1).enumerate() {
                let placeholder = format!("{{{}}}", i);
                let value = arr.value(row);
                prompt = prompt.replace(&placeholder, value);
            }

            prompts.push(prompt);
        }

        // Filter out null entries for the API call
        let valid_prompts: Vec<String> = prompts
            .iter()
            .enumerate()
            .filter(|(i, _)| !null_indices.contains(i))
            .map(|(_, p)| p.clone())
            .collect();

        // Call the batch API
        let client = self.client.clone();
        let api_results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { client.process_prompts(valid_prompts).await })
        })
        .map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!("LLM API error: {}", e))
        })?;

        // Reconstruct results with nulls in correct positions
        let mut result_builder = arrow::array::StringBuilder::new();
        let mut api_idx = 0;

        for i in 0..num_rows {
            if null_indices.contains(&i) {
                result_builder.append_null();
            } else {
                result_builder.append_value(&api_results[api_idx]);
                api_idx += 1;
            }
        }

        let result_array: ArrayRef = Arc::new(result_builder.finish());
        Ok(ColumnarValue::Array(result_array))
    }
}

impl ScalarUDFImpl for LlmUnfoldUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "llm_unfold"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        // Return List<Utf8>
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        ))))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let args = &args.args;
        if args.len() < 3 {
            return Err(datafusion::error::DataFusionError::Execution(
                "llm_unfold() requires: (template, column, delimiter)".to_string(),
            ));
        }

        // Args: template, column, delimiter
        let num_rows = args
            .iter()
            .find_map(|a| match a {
                ColumnarValue::Array(arr) => Some(arr.len()),
                _ => None,
            })
            .unwrap_or(1);

        // Convert all arguments to arrays
        let arrays: Vec<ArrayRef> = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(arr) => Ok(arr.clone()),
                ColumnarValue::Scalar(s) => s.to_array_of_size(num_rows),
            })
            .collect::<DFResult<Vec<_>>>()?;

        // Convert to string arrays (supports both Utf8 and Utf8View)
        let string_arrays: Vec<StringArrayRef> = arrays
            .iter()
            .map(|arr| {
                StringArrayRef::try_from_array(arr).ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "All arguments to llm_unfold() must be strings".to_string(),
                    )
                })
            })
            .collect::<DFResult<Vec<_>>>()?;

        // Get template (should be constant across rows, use first row)
        let template = string_arrays[0].value(0);
        let delimiter = string_arrays[2].value(0);

        // Parse template to detect range syntax for batching
        let parsed = parse_template(template);

        // Determine batch size from range syntax
        // If template has {0:9}, batch_size = 10
        // If template has just {0}, batch_size = 1 (per-row)
        let batch_size = if !parsed.ranges.is_empty() {
            // Use the first range to determine batch size
            let range = &parsed.ranges[0];
            range.end - range.start + 1
        } else if let Some(max) = parsed.max_placeholder {
            // Simple placeholders like {0}, {1}, {2} → batch_size = max + 1
            max + 1
        } else {
            1
        };

        // Collect column values (second argument)
        let column_values: Vec<String> = (0..num_rows)
            .map(|i| string_arrays[1].value(i).to_string())
            .collect();

        // Process in batches
        let num_batches = (num_rows + batch_size - 1) / batch_size;
        let mut prompts = Vec::with_capacity(num_batches);
        let mut batch_sizes_actual = Vec::with_capacity(num_batches); // Track actual size of each batch

        for batch_idx in 0..num_batches {
            let start_row = batch_idx * batch_size;
            let end_row = (start_row + batch_size).min(num_rows);
            let actual_batch_size = end_row - start_row;
            batch_sizes_actual.push(actual_batch_size);

            // Collect values for this batch
            let batch_values: Vec<&str> = column_values[start_row..end_row]
                .iter()
                .map(|s| s.as_str())
                .collect();

            // If batch is smaller than expected (last batch), pad with empty strings
            let mut padded_values = batch_values.clone();
            while padded_values.len() < batch_size {
                padded_values.push("");
            }

            // Build prompt using expand_template
            let prompt = expand_template(template, &padded_values).map_err(|e| {
                datafusion::error::DataFusionError::Execution(format!("Template error: {}", e))
            })?;

            prompts.push(prompt);
        }

        // Call the batch API
        let client = self.client.clone();
        let api_results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { client.process_prompts(prompts).await })
        })
        .map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!("LLM API error: {}", e))
        })?;

        // Build list array with split results distributed back to rows
        let mut list_builder = GenericListBuilder::<i32, StringBuilder>::new(StringBuilder::new());

        let mut row_idx = 0;
        for (batch_idx, result) in api_results.iter().enumerate() {
            let actual_batch_size = batch_sizes_actual[batch_idx];

            // Split the result by delimiter
            let items: Vec<&str> = if delimiter.is_empty() {
                vec![result.as_str()]
            } else {
                result
                    .split(delimiter)
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect()
            };

            // Distribute items back to rows in this batch
            for i in 0..actual_batch_size {
                // Each row gets its corresponding item (or empty if not enough items)
                let item = items.get(i).copied().unwrap_or("");
                list_builder.values().append_value(item);
                list_builder.append(true);
                row_idx += 1;
            }
        }

        // Handle any remaining rows (shouldn't happen, but safety)
        while row_idx < num_rows {
            list_builder.append_null();
            row_idx += 1;
        }

        let result_array: ArrayRef = Arc::new(list_builder.finish());
        Ok(ColumnarValue::Array(result_array))
    }
}
