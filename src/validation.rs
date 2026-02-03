use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TemplateError {
    #[error("Missing placeholder {{{0}}} in template (have {1} arguments)")]
    MissingPlaceholder(usize, usize),
    #[error("Unused argument at position {0} (template only uses placeholders up to {{{1}}})")]
    UnusedArgument(usize, usize),
    #[error("Unclosed placeholder starting at position {0}")]
    UnclosedPlaceholder(usize),
    #[error("Invalid placeholder '{{}}' at position {0} (expected a number)")]
    InvalidPlaceholder(usize),
    #[error("Placeholder {{{0}}} exceeds maximum index {1}")]
    PlaceholderOutOfRange(usize, usize),
    #[error("Invalid range placeholder at position {0}: {1}")]
    InvalidRange(usize, String),
}

/// Represents a range placeholder like {0:4, } which expands to {0}, {1}, {2}, {3}, {4}
#[derive(Debug, Clone)]
pub struct RangePlaceholder {
    pub start: usize,
    pub end: usize, // inclusive
    pub separator: String,
    pub position: usize, // position in template string
}

#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub errors: Vec<TemplateError>,
    pub warnings: Vec<String>,
    pub placeholders_found: HashSet<usize>,
    pub max_placeholder: Option<usize>,
    pub ranges: Vec<RangePlaceholder>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Parse a template and extract all placeholders {0}, {1}, etc.
/// Also supports range placeholders like {0:4, } which expand to {0}, {1}, {2}, {3}, {4}
/// Range syntax: {start:end<separator>} where end is inclusive
pub fn parse_template(template: &str) -> ValidationResult {
    let mut result = ValidationResult::default();
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            // Check for escaped brace {{
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                i += 2;
                continue;
            }

            let start = i;
            i += 1;

            // Find closing brace
            let mut content = String::new();
            while i < chars.len() && chars[i] != '}' {
                content.push(chars[i]);
                i += 1;
            }

            if i >= chars.len() {
                result
                    .errors
                    .push(TemplateError::UnclosedPlaceholder(start));
                break;
            }

            // Parse the content - could be simple {N} or range {N:M<sep>}
            if content.is_empty() {
                result.errors.push(TemplateError::InvalidPlaceholder(start));
            } else if let Some(colon_pos) = content.find(':') {
                // Range placeholder: {start:end<separator>}
                let start_str = &content[..colon_pos];
                let rest = &content[colon_pos + 1..];

                // Parse start number
                let range_start = match start_str.parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => {
                        result.errors.push(TemplateError::InvalidRange(
                            start,
                            format!("invalid start index '{}'", start_str),
                        ));
                        i += 1;
                        continue;
                    }
                };

                // Find where the end number stops (first non-digit)
                let end_num_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
                if end_num_len == 0 {
                    result.errors.push(TemplateError::InvalidRange(
                        start,
                        "missing end index after ':'".to_string(),
                    ));
                    i += 1;
                    continue;
                }

                let end_str = &rest[..end_num_len];
                let separator = rest[end_num_len..].to_string();

                let range_end = match end_str.parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => {
                        result.errors.push(TemplateError::InvalidRange(
                            start,
                            format!("invalid end index '{}'", end_str),
                        ));
                        i += 1;
                        continue;
                    }
                };

                if range_start > range_end {
                    result.errors.push(TemplateError::InvalidRange(
                        start,
                        format!("start {} > end {}", range_start, range_end),
                    ));
                    i += 1;
                    continue;
                }

                // Add all indices in range to placeholders_found
                for idx in range_start..=range_end {
                    result.placeholders_found.insert(idx);
                    result.max_placeholder =
                        Some(result.max_placeholder.map_or(idx, |max| max.max(idx)));
                }

                result.ranges.push(RangePlaceholder {
                    start: range_start,
                    end: range_end,
                    separator,
                    position: start,
                });
            } else if let Ok(n) = content.parse::<usize>() {
                result.placeholders_found.insert(n);
                result.max_placeholder = Some(result.max_placeholder.map_or(n, |max| max.max(n)));
            } else {
                // Not a number - might be a different kind of placeholder, just warn
                result.warnings.push(format!(
                    "Non-numeric placeholder '{{{}}}' at position {}",
                    content, start
                ));
            }

            i += 1; // Skip closing brace
        } else if chars[i] == '}' {
            // Check for escaped brace }}
            if i + 1 < chars.len() && chars[i + 1] == '}' {
                i += 2;
                continue;
            }
            // Lone closing brace - might be intentional, just warn
            result
                .warnings
                .push(format!("Unmatched '}}' at position {}", i));
            i += 1;
        } else {
            i += 1;
        }
    }

    result
}

/// Validate a template against expected argument count
/// Returns Ok(warnings) if valid, or Err(error_message) if invalid
///
/// In non-strict mode, missing placeholders are errors if ALL placeholders are missing
/// (likely wrong argument order), but individual missing placeholders are warnings.
pub fn validate_template(
    template: &str,
    arg_count: usize,
    strict: bool,
) -> Result<Vec<String>, String> {
    let result = parse_template(template);
    let mut warnings = result.warnings.clone();

    // Check for parse errors
    if !result.errors.is_empty() {
        return Err(result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; "));
    }

    if arg_count == 0 {
        if !result.placeholders_found.is_empty() {
            return Err(format!(
                "Template has placeholders but no arguments provided"
            ));
        }
        return Ok(warnings);
    }

    // Check if template has NO placeholders at all but arguments were provided
    // This is likely a wrong argument order error - always error
    if result.placeholders_found.is_empty() {
        return Err(format!(
            "Template has no placeholders ({{0}}, {{1}}, etc.) but {} argument(s) provided. \
             Did you put the template in the wrong position? \
             Usage: llm('template with {{0}}', arg0, arg1, ...)",
            arg_count
        ));
    }

    // Check that all expected placeholders {0} through {arg_count-1} exist
    for i in 0..arg_count {
        if !result.placeholders_found.contains(&i) {
            let msg = format!("Argument {} is not used in template (missing {{{}}})", i, i);
            if strict {
                return Err(msg);
            } else {
                warnings.push(msg);
            }
        }
    }

    // Check for placeholders beyond arg_count
    if let Some(max) = result.max_placeholder {
        if max >= arg_count {
            return Err(format!(
                "Template references {{{}}} but only {} argument(s) provided (indices 0-{})",
                max,
                arg_count,
                arg_count.saturating_sub(1)
            ));
        }
    }

    Ok(warnings)
}

/// Expand a template by substituting values for placeholders
/// Handles both simple {N} placeholders and range {N:M<sep>} placeholders
pub fn expand_template(template: &str, values: &[&str]) -> Result<String, String> {
    let result = parse_template(template);

    if !result.errors.is_empty() {
        return Err(result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; "));
    }

    // Check that all referenced placeholders have values
    if let Some(max) = result.max_placeholder {
        if max >= values.len() {
            return Err(format!(
                "Template references {{{}}} but only {} value(s) provided",
                max,
                values.len()
            ));
        }
    }

    let mut output = String::new();
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            // Check for escaped brace {{
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                output.push('{');
                i += 2;
                continue;
            }

            i += 1;

            // Find closing brace
            let mut content = String::new();
            while i < chars.len() && chars[i] != '}' {
                content.push(chars[i]);
                i += 1;
            }

            // Parse and substitute
            if let Some(colon_pos) = content.find(':') {
                // Range placeholder
                let start_str = &content[..colon_pos];
                let rest = &content[colon_pos + 1..];
                let end_num_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
                let end_str = &rest[..end_num_len];
                let separator = &rest[end_num_len..];

                let range_start: usize = start_str.parse().unwrap();
                let range_end: usize = end_str.parse().unwrap();

                let expanded: Vec<&str> =
                    (range_start..=range_end).map(|idx| values[idx]).collect();
                output.push_str(&expanded.join(separator));
            } else if let Ok(n) = content.parse::<usize>() {
                output.push_str(values[n]);
            }

            i += 1; // Skip closing brace
        } else if chars[i] == '}' {
            // Check for escaped brace }}
            if i + 1 < chars.len() && chars[i + 1] == '}' {
                output.push('}');
                i += 2;
                continue;
            }
            output.push(chars[i]);
            i += 1;
        } else {
            output.push(chars[i]);
            i += 1;
        }
    }

    Ok(output)
}

/// Validate the fold template (expects {0} and {1}) - legacy 2-way
pub fn validate_fold_template(template: &str) -> Result<Vec<String>, String> {
    validate_reduce_template(template).map(|(warnings, _)| warnings)
}

/// Validate reduce template and return (warnings, arity K)
/// Arity is determined by the number of sequential placeholders {0}, {1}, ..., {K-1}
/// Must have at least {0} and {1} (K >= 2)
pub fn validate_reduce_template(template: &str) -> Result<(Vec<String>, usize), String> {
    let result = parse_template(template);
    let warnings = result.warnings.clone();

    if !result.errors.is_empty() {
        return Err(result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; "));
    }

    // Must have at least {0} and {1}
    if !result.placeholders_found.contains(&0) {
        return Err("Reduce template must contain {0}".to_string());
    }
    if !result.placeholders_found.contains(&1) {
        return Err(
            "Reduce template must contain {1} (need at least 2 placeholders for reduce)"
                .to_string(),
        );
    }

    // Determine arity K by finding the highest sequential placeholder
    // {0}, {1}, {2} -> K=3, but {0}, {1}, {5} -> K=2 (gap detected)
    let mut k = 0usize;
    while result.placeholders_found.contains(&k) {
        k += 1;
    }

    // Warn about gaps (e.g., {0}, {1}, {5} but missing {2}, {3}, {4})
    if let Some(max) = result.max_placeholder {
        if max >= k {
            return Err(format!(
                "Reduce template has gap in placeholders: found {{{}}} but missing {{{}}}",
                max, k
            ));
        }
    }

    Ok((warnings, k))
}

/// Validate the map template (expects {0})
pub fn validate_map_template(template: &str) -> Result<Vec<String>, String> {
    let result = parse_template(template);
    let mut warnings = result.warnings.clone();

    if !result.errors.is_empty() {
        return Err(result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; "));
    }

    // Map template must have {0}
    if !result.placeholders_found.contains(&0) {
        return Err("Map template must contain {0} for the input value".to_string());
    }

    // Warn about extra placeholders
    if let Some(max) = result.max_placeholder {
        if max > 0 {
            warnings.push(format!(
                "Map template has placeholder {{{}}} but only {{0}} is used",
                max
            ));
        }
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_template() {
        assert!(validate_template("Hello {0}!", 1, true).is_ok());
        assert!(validate_template("Translate {0} to {1}", 2, true).is_ok());
        assert!(validate_template("{0} + {1} = {2}", 3, true).is_ok());
    }

    #[test]
    fn test_missing_placeholder() {
        // Strict mode: error
        assert!(validate_template("Hello {0}!", 2, true).is_err());
        // Non-strict: warning (but still valid since {0} exists)
        let result = validate_template("Hello {0}!", 2, false).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_no_placeholders_with_args() {
        // Template with no placeholders but args provided - always error
        assert!(validate_template("Hello world!", 1, false).is_err());
        assert!(validate_template("Hello world!", 2, true).is_err());
    }

    #[test]
    fn test_placeholder_out_of_range() {
        assert!(validate_template("Hello {5}!", 2, true).is_err());
    }

    #[test]
    fn test_unclosed_placeholder() {
        assert!(validate_template("Hello {0!", 1, true).is_err());
    }

    #[test]
    fn test_escaped_braces() {
        assert!(validate_template("Use {{0}} for literal braces", 0, true).is_ok());
    }

    #[test]
    fn test_no_args_no_placeholders() {
        // No args, no placeholders - valid
        assert!(validate_template("Just a static prompt", 0, true).is_ok());
    }

    #[test]
    fn test_fold_template() {
        assert!(validate_fold_template("Combine: {0} and {1}").is_ok());
        assert!(validate_fold_template("Only {0}").is_err());
        assert!(validate_fold_template("Only {1}").is_err());
    }

    #[test]
    fn test_map_template() {
        assert!(validate_map_template("Process: {0}").is_ok());
        assert!(validate_map_template("No placeholder").is_err());
    }

    #[test]
    fn test_range_placeholder_parse() {
        let result = parse_template("Combine: {0:2, }");
        assert!(result.is_valid());
        assert!(result.placeholders_found.contains(&0));
        assert!(result.placeholders_found.contains(&1));
        assert!(result.placeholders_found.contains(&2));
        assert_eq!(result.max_placeholder, Some(2));
        assert_eq!(result.ranges.len(), 1);
        assert_eq!(result.ranges[0].start, 0);
        assert_eq!(result.ranges[0].end, 2);
        assert_eq!(result.ranges[0].separator, ", ");
    }

    #[test]
    fn test_range_placeholder_expand() {
        let result = expand_template("Items: {0:2, }", &["a", "b", "c"]).unwrap();
        assert_eq!(result, "Items: a, b, c");

        let result = expand_template("List:\n{0:3\n}", &["one", "two", "three", "four"]).unwrap();
        assert_eq!(result, "List:\none\ntwo\nthree\nfour");
    }

    #[test]
    fn test_mixed_placeholders() {
        let result = expand_template("First: {0}, Rest: {1:3, }", &["a", "b", "c", "d"]).unwrap();
        assert_eq!(result, "First: a, Rest: b, c, d");
    }

    #[test]
    fn test_k_way_reduce_template() {
        // 3-way reduce
        let (_, k) = validate_reduce_template("Combine: {0:2, }").unwrap();
        assert_eq!(k, 3);

        // 4-way reduce
        let (_, k) = validate_reduce_template("{0} + {1} + {2} + {3}").unwrap();
        assert_eq!(k, 4);

        // Range covers 0-4, so K=5
        let (_, k) = validate_reduce_template("Merge all: {0:4\n---\n}").unwrap();
        assert_eq!(k, 5);
    }
}
