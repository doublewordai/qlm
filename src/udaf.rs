use arrow::array::{Array, ArrayRef, StringArray, StringViewArray};
use arrow::datatypes::{DataType, Field};
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDFImpl, Signature, TypeSignature, Volatility,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::any::Any;
use std::time::Duration;

use crate::client::LlmClient;
use crate::validation::{expand_template, validate_reduce_template};

/// Helper to extract string values from either StringArray or StringViewArray
fn get_string_value(arr: &ArrayRef, i: usize) -> Option<&str> {
    if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
        if s.is_null(i) {
            None
        } else {
            Some(s.value(i))
        }
    } else if let Some(s) = arr.as_any().downcast_ref::<StringViewArray>() {
        if s.is_null(i) {
            None
        } else {
            Some(s.value(i))
        }
    } else {
        None
    }
}

fn get_string_array_len(arr: &ArrayRef) -> Option<usize> {
    if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
        Some(s.len())
    } else if let Some(s) = arr.as_any().downcast_ref::<StringViewArray>() {
        Some(s.len())
    } else {
        None
    }
}

/// LLM Fold: aggregate M rows → 1 output via K-way tree-reduce
///
/// Forms:
///   llm_fold(column, reduce_prompt)              -- fold only
///   llm_fold(column, reduce_prompt, map_prompt)  -- map then fold
///
/// K is auto-detected from placeholders. Use {0:K-1<sep>} for K-way folding.
///
/// Examples:
///   SELECT llm_fold(text, 'Combine:\n{0}\n---\n{1}') FROM docs;          -- 2-way
///   SELECT llm_fold(text, 'Merge:\n{0:3\n---\n}') FROM docs;             -- 4-way
///   SELECT llm_fold(text, 'Combine:\n{0}\n{1}', 'Summarize: {0}') FROM docs;
#[derive(Debug)]
pub struct LlmFoldUdaf {
    signature: Signature,
    client: LlmClient,
}

impl LlmFoldUdaf {
    pub fn new(client: LlmClient) -> Self {
        Self {
            // Accept 2 args (reduce only) or 3 args (map + reduce)
            // Support both Utf8 and Utf8View for Parquet compatibility
            signature: Signature::new(TypeSignature::VariadicAny, Volatility::Volatile),
            client,
        }
    }
}

impl AggregateUDFImpl for LlmFoldUdaf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "llm_fold"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn accumulator(&self, args: AccumulatorArgs) -> DFResult<Box<dyn Accumulator>> {
        let has_map = args.exprs.len() == 3;
        Ok(Box::new(LlmAggAccumulator::new(
            self.client.clone(),
            has_map,
        )))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> DFResult<Vec<Field>> {
        Ok(vec![
            Field::new("reduce_prompt", DataType::Utf8, true),
            Field::new("map_prompt", DataType::Utf8, true),
            Field::new("values", DataType::Utf8, true), // JSON array
        ])
    }
}

#[derive(Debug)]
struct LlmAggAccumulator {
    client: LlmClient,
    reduce_prompt: Option<String>,
    map_prompt: Option<String>,
    values: Vec<String>,
    has_map: bool,
}

impl LlmAggAccumulator {
    fn new(client: LlmClient, has_map: bool) -> Self {
        Self {
            client,
            reduce_prompt: None,
            map_prompt: None,
            values: Vec::new(),
            has_map,
        }
    }

    fn tree_reduce(
        &self,
        mut items: Vec<String>,
        progress: &ProgressBar,
        k: usize,
    ) -> DFResult<String> {
        let reduce_prompt = self.reduce_prompt.as_ref().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution("Missing reduce prompt".to_string())
        })?;

        if items.is_empty() {
            return Ok(String::new());
        }

        if items.len() == 1 {
            return Ok(items.remove(0));
        }

        let client = self.client.clone();
        let template = reduce_prompt.clone();

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut level = 1u64;
                while items.len() > 1 {
                    let mut prompts = Vec::new();
                    let mut new_items = Vec::new();

                    let mut i = 0;
                    while i < items.len() {
                        // How many items can we group? At least 2, up to K
                        let remaining = items.len() - i;
                        if remaining >= k {
                            // Full K-way group
                            let group: Vec<&str> =
                                items[i..i + k].iter().map(|s| s.as_str()).collect();
                            let prompt = expand_template(&template, &group).map_err(|e| {
                                datafusion::error::DataFusionError::Execution(format!(
                                    "Template expansion error: {}",
                                    e
                                ))
                            })?;
                            prompts.push(prompt);
                            i += k;
                        } else if remaining >= 2 {
                            // Partial group (at least 2 items) - still reduce them
                            // Build a partial prompt using available items
                            let group: Vec<&str> = items[i..].iter().map(|s| s.as_str()).collect();
                            // For partial groups, we need to handle the template carefully
                            // We'll pad with empty strings if needed (though this may not be ideal)
                            let mut padded = group.clone();
                            while padded.len() < k {
                                padded.push("");
                            }
                            let prompt = expand_template(&template, &padded).map_err(|e| {
                                datafusion::error::DataFusionError::Execution(format!(
                                    "Template expansion error: {}",
                                    e
                                ))
                            })?;
                            prompts.push(prompt);
                            i = items.len();
                        } else {
                            // Only 1 item left, carry forward
                            new_items.push(items[i].clone());
                            i += 1;
                        }
                    }

                    if !prompts.is_empty() {
                        progress.set_message(format!(
                            "{}-way reduce level {} ({} → {})",
                            k,
                            level,
                            items.len(),
                            prompts.len() + new_items.len()
                        ));
                        let results = client.process_prompts_quiet(prompts).await.map_err(|e| {
                            datafusion::error::DataFusionError::Execution(format!(
                                "LLM API error during reduce: {}",
                                e
                            ))
                        })?;
                        new_items.extend(results);
                    }

                    items = new_items;
                    level += 1;
                }

                Ok(items.remove(0))
            })
        })
    }
}

impl Accumulator for LlmAggAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DFResult<()> {
        // values[0] = column, values[1] = reduce_prompt, values[2] = map_prompt (optional)
        let content_len = get_string_array_len(&values[0]).ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "First argument (column) must be string".to_string(),
            )
        })?;

        let reduce_len = get_string_array_len(&values[1]).ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "Second argument (reduce_prompt) must be string".to_string(),
            )
        })?;

        // Capture reduce prompt from first non-null
        for i in 0..reduce_len {
            if self.reduce_prompt.is_none() {
                if let Some(v) = get_string_value(&values[1], i) {
                    self.reduce_prompt = Some(v.to_string());
                    break;
                }
            }
        }

        // Capture map prompt if provided
        if self.has_map && values.len() > 2 {
            if let Some(map_len) = get_string_array_len(&values[2]) {
                for i in 0..map_len {
                    if self.map_prompt.is_none() {
                        if let Some(v) = get_string_value(&values[2], i) {
                            self.map_prompt = Some(v.to_string());
                            break;
                        }
                    }
                }
            }
        }

        // Collect values
        for i in 0..content_len {
            if let Some(v) = get_string_value(&values[0], i) {
                self.values.push(v.to_string());
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> DFResult<ScalarValue> {
        if self.values.is_empty() {
            return Ok(ScalarValue::Utf8(None));
        }

        let reduce_prompt = self.reduce_prompt.as_ref().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution("Missing reduce prompt".to_string())
        })?;

        // Validate reduce prompt and get arity K
        let k = match validate_reduce_template(reduce_prompt) {
            Ok((warnings, k)) => {
                for w in warnings {
                    eprintln!("Warning: {}", w);
                }
                k
            }
            Err(e) => {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Invalid reduce prompt: {}. Must contain at least {{0}} and {{1}}.",
                    e
                )));
            }
        };

        // Validate map prompt if provided
        if let Some(ref map_prompt) = self.map_prompt {
            if !map_prompt.contains("{0}") {
                return Err(datafusion::error::DataFusionError::Execution(
                    "Map prompt must contain {0} placeholder".to_string(),
                ));
            }
        }

        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.cyan} llm_fold: {msg}")
                .unwrap(),
        );
        progress.enable_steady_tick(Duration::from_millis(80));

        let item_count = self.values.len();

        // Step 1: Map if map_prompt provided, otherwise use raw values
        let items = if let Some(ref map_prompt) = self.map_prompt {
            progress.set_message(format!("mapping {} items...", item_count));

            let prompts: Vec<String> = self
                .values
                .iter()
                .map(|v| map_prompt.replace("{0}", v))
                .collect();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async { self.client.process_prompts_quiet(prompts).await })
            })
            .map_err(|e| {
                progress.finish_with_message(format!("✗ map failed: {}", e));
                datafusion::error::DataFusionError::Execution(format!(
                    "LLM API error during map: {}",
                    e
                ))
            })?
        } else {
            progress.set_message(format!("{}-way reducing {} items...", k, item_count));
            self.values.clone()
        };

        // Step 2: Tree-reduce with K-way folding
        let result = self.tree_reduce(items, &progress, k)?;

        progress.finish_with_message(format!("✓ complete ({} items, {}-way)", item_count, k));

        Ok(ScalarValue::Utf8(Some(result)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.values.iter().map(|s| s.len()).sum::<usize>()
            + self.reduce_prompt.as_ref().map(|s| s.len()).unwrap_or(0)
            + self.map_prompt.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    fn state(&mut self) -> DFResult<Vec<ScalarValue>> {
        let values_json = serde_json::to_string(&self.values).unwrap_or_default();
        Ok(vec![
            ScalarValue::Utf8(self.reduce_prompt.clone()),
            ScalarValue::Utf8(self.map_prompt.clone()),
            ScalarValue::Utf8(Some(values_json)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DFResult<()> {
        if states.len() != 3 {
            return Err(datafusion::error::DataFusionError::Execution(
                "Invalid state for llm_fold".to_string(),
            ));
        }

        let reduce_prompts = states[0].as_any().downcast_ref::<StringArray>().unwrap();
        let map_prompts = states[1].as_any().downcast_ref::<StringArray>().unwrap();
        let values_jsons = states[2].as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..reduce_prompts.len() {
            if self.reduce_prompt.is_none() && !reduce_prompts.is_null(i) {
                self.reduce_prompt = Some(reduce_prompts.value(i).to_string());
            }
            if self.map_prompt.is_none() && !map_prompts.is_null(i) {
                self.map_prompt = Some(map_prompts.value(i).to_string());
            }
            if !values_jsons.is_null(i) {
                let json = values_jsons.value(i);
                if let Ok(vals) = serde_json::from_str::<Vec<String>>(json) {
                    self.values.extend(vals);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use std::sync::Arc;

    #[test]
    fn test_get_string_value_from_string_array() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["test", "value"]));

        assert_eq!(get_string_value(&arr, 0), Some("test"));
        assert_eq!(get_string_value(&arr, 1), Some("value"));
    }

    #[test]
    fn test_get_string_value_from_string_view_array() {
        let arr: ArrayRef = Arc::new(StringViewArray::from(vec!["view1", "view2"]));

        assert_eq!(get_string_value(&arr, 0), Some("view1"));
        assert_eq!(get_string_value(&arr, 1), Some("view2"));
    }

    #[test]
    fn test_get_string_value_null() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));

        assert_eq!(get_string_value(&arr, 0), Some("a"));
        assert_eq!(get_string_value(&arr, 1), None);
        assert_eq!(get_string_value(&arr, 2), Some("c"));
    }

    #[test]
    fn test_get_string_value_invalid_type() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        assert_eq!(get_string_value(&arr, 0), None);
    }

    #[test]
    fn test_get_string_array_len_string_array() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        assert_eq!(get_string_array_len(&arr), Some(3));
    }

    #[test]
    fn test_get_string_array_len_string_view_array() {
        let arr: ArrayRef = Arc::new(StringViewArray::from(vec!["x", "y"]));
        assert_eq!(get_string_array_len(&arr), Some(2));
    }

    #[test]
    fn test_get_string_array_len_empty() {
        let arr: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
        assert_eq!(get_string_array_len(&arr), Some(0));
    }

    #[test]
    fn test_get_string_array_len_invalid_type() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        assert_eq!(get_string_array_len(&arr), None);
    }

    #[test]
    fn test_llm_fold_udaf_name() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let udaf = LlmFoldUdaf::new(client);
        assert_eq!(udaf.name(), "llm_fold");
    }

    #[test]
    fn test_llm_fold_udaf_return_type() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let udaf = LlmFoldUdaf::new(client);
        let return_type = udaf.return_type(&[]).unwrap();
        assert_eq!(return_type, DataType::Utf8);
    }

    #[test]
    fn test_llm_fold_udaf_signature_variadic() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let udaf = LlmFoldUdaf::new(client);
        let sig = udaf.signature();
        assert_eq!(sig.volatility, Volatility::Volatile);
    }

    #[test]
    fn test_accumulator_initial_state() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let acc = LlmAggAccumulator::new(client, false);

        assert!(acc.reduce_prompt.is_none());
        assert!(acc.map_prompt.is_none());
        assert!(acc.values.is_empty());
        assert!(!acc.has_map);
    }

    #[test]
    fn test_accumulator_initial_state_with_map() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let acc = LlmAggAccumulator::new(client, true);

        assert!(acc.has_map);
    }

    #[test]
    fn test_accumulator_size() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, false);

        let initial_size = acc.size();

        acc.values.push("test value 1".to_string());
        acc.values.push("test value 2".to_string());
        acc.reduce_prompt = Some("reduce: {0} {1}".to_string());

        let final_size = acc.size();
        assert!(final_size > initial_size);
    }

    #[test]
    fn test_accumulator_state_serialization() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, true);

        acc.reduce_prompt = Some("reduce {0} {1}".to_string());
        acc.map_prompt = Some("map {0}".to_string());
        acc.values = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let state = acc.state().unwrap();
        assert_eq!(state.len(), 3);

        // Verify reduce_prompt
        match &state[0] {
            ScalarValue::Utf8(Some(s)) => assert_eq!(s, "reduce {0} {1}"),
            _ => panic!("Expected Utf8 for reduce_prompt"),
        }

        // Verify map_prompt
        match &state[1] {
            ScalarValue::Utf8(Some(s)) => assert_eq!(s, "map {0}"),
            _ => panic!("Expected Utf8 for map_prompt"),
        }

        // Verify values JSON
        match &state[2] {
            ScalarValue::Utf8(Some(json)) => {
                let values: Vec<String> = serde_json::from_str(json).unwrap();
                assert_eq!(values, vec!["a", "b", "c"]);
            }
            _ => panic!("Expected Utf8 JSON for values"),
        }
    }

    #[test]
    fn test_accumulator_state_with_none_prompts() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, false);

        acc.values = vec!["value1".to_string()];

        let state = acc.state().unwrap();

        // reduce_prompt should be None
        assert!(matches!(&state[0], ScalarValue::Utf8(None)));

        // map_prompt should be None
        assert!(matches!(&state[1], ScalarValue::Utf8(None)));
    }

    #[test]
    fn test_accumulator_evaluate_empty_values() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, false);

        let result = acc.evaluate().unwrap();
        assert!(matches!(result, ScalarValue::Utf8(None)));
    }

    #[test]
    fn test_accumulator_evaluate_missing_reduce_prompt() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, false);
        acc.values = vec!["value".to_string()];

        let result = acc.evaluate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("reduce prompt"));
    }

    #[test]
    fn test_accumulator_evaluate_invalid_reduce_prompt() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let mut acc = LlmAggAccumulator::new(client, false);
        acc.values = vec!["value".to_string()];
        acc.reduce_prompt = Some("Only {0}".to_string()); // Missing {1}

        let result = acc.evaluate();
        assert!(result.is_err());
    }

    #[test]
    fn test_state_fields() {
        let client = crate::client::LlmClient::new("http://test", "key", "model");
        let udaf = LlmFoldUdaf::new(client);

        let fields = udaf
            .state_fields(StateFieldsArgs {
                name: "test",
                input_types: &[],
                return_type: &DataType::Utf8,
                ordering_fields: &[],
                is_distinct: false,
            })
            .unwrap();

        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), "reduce_prompt");
        assert_eq!(fields[1].name(), "map_prompt");
        assert_eq!(fields[2].name(), "values");

        for field in &fields {
            assert_eq!(field.data_type(), &DataType::Utf8);
        }
    }

    #[test]
    fn test_get_string_value_unicode() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["日本語", "emoji 🎉", "مرحبا"]));

        assert_eq!(get_string_value(&arr, 0), Some("日本語"));
        assert_eq!(get_string_value(&arr, 1), Some("emoji 🎉"));
        assert_eq!(get_string_value(&arr, 2), Some("مرحبا"));
    }

    #[test]
    fn test_get_string_value_special_chars() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            "line1\nline2",
            "tab\there",
            "quote\"test",
        ]));

        assert_eq!(get_string_value(&arr, 0), Some("line1\nline2"));
        assert_eq!(get_string_value(&arr, 1), Some("tab\there"));
        assert_eq!(get_string_value(&arr, 2), Some("quote\"test"));
    }

    // Note: Integration tests for update_batch, merge_batch, and tree_reduce
    // would require mocking the LLM client, which is better done in integration tests
}
