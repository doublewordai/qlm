//! End-to-end CLI tests for qlm
//!
//! These tests verify the CLI interface works correctly.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

fn qlm_cmd() -> Command {
    Command::cargo_bin("qlm").unwrap()
}

#[test]
fn test_help_flag() {
    qlm_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("SQL shell"))
        .stdout(predicate::str::contains("LLM"));
}

#[test]
fn test_version_flag() {
    qlm_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("qlm"));
}

#[test]
fn test_missing_api_key() {
    // When no API key is provided, the tool should fail with a helpful message
    qlm_cmd()
        .env_remove("DOUBLEWORD_API_KEY")
        .arg("-c")
        .arg("SELECT 1")
        .assert()
        .failure();
}

#[test]
fn test_load_csv_file_basic_query() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "id,name,value").unwrap();
    writeln!(csv_file, "1,test,100").unwrap();
    writeln!(csv_file, "2,foo,200").unwrap();
    writeln!(csv_file, "3,bar,300").unwrap();

    // Test that we can load a CSV and run a simple query (no LLM required)
    // Use explicit table name to avoid temp file naming issues
    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", csv_file.path().display()))
        .arg("-c")
        .arg("SELECT COUNT(*) as cnt FROM data;")
        .assert()
        .success();
}

#[test]
fn test_load_table_with_custom_name() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "col1,col2").unwrap();
    writeln!(csv_file, "a,1").unwrap();
    writeln!(csv_file, "b,2").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("my_table={}", csv_file.path().display()))
        .arg("-c")
        .arg("SELECT * FROM my_table;")
        .assert()
        .success();
}

#[test]
fn test_sql_syntax_error() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "x,y").unwrap();
    writeln!(csv_file, "1,2").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", csv_file.path().display()))
        .arg("-c")
        .arg("SELEC * FORM data;") // Intentional syntax errors
        .assert()
        .failure();
}

#[test]
fn test_nonexistent_table() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "a,b").unwrap();
    writeln!(csv_file, "1,2").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", csv_file.path().display()))
        .arg("-c")
        .arg("SELECT * FROM nonexistent_table;")
        .assert()
        .failure();
}

#[test]
fn test_sql_file_execution() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "x,y").unwrap();
    writeln!(csv_file, "1,2").unwrap();
    writeln!(csv_file, "3,4").unwrap();

    let mut sql_file = NamedTempFile::with_suffix(".sql").unwrap();
    writeln!(sql_file, "SELECT x + y as sum FROM data;").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", csv_file.path().display()))
        .arg("-f")
        .arg(sql_file.path())
        .assert()
        .success();
}

#[test]
fn test_unsupported_file_format() {
    let mut file = NamedTempFile::with_suffix(".xyz").unwrap();
    writeln!(file, "some data").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(file.path())
        .assert()
        .failure();
}

#[test]
fn test_multiple_tables() {
    let mut csv1 = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv1, "id,name").unwrap();
    writeln!(csv1, "1,Alice").unwrap();

    let mut csv2 = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv2, "id,score").unwrap();
    writeln!(csv2, "1,100").unwrap();

    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("users={}", csv1.path().display()))
        .arg("-t")
        .arg(format!("scores={}", csv2.path().display()))
        .arg("-c")
        .arg("SELECT u.name, s.score FROM users u JOIN scores s ON u.id = s.id;")
        .assert()
        .success();
}

#[test]
fn test_json_file_loading() {
    let mut json_file = NamedTempFile::with_suffix(".json").unwrap();
    writeln!(json_file, r#"{{"id": 1, "value": "test"}}"#).unwrap();
    writeln!(json_file, r#"{{"id": 2, "value": "data"}}"#).unwrap();

    // Use explicit table name for JSON files
    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", json_file.path().display()))
        .arg("-c")
        .arg("SELECT COUNT(*) FROM data;")
        .assert()
        .success();
}

#[test]
fn test_empty_command() {
    // Empty SQL command should fail with an appropriate error
    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-c")
        .arg("")
        .assert()
        .failure()
        .stderr(predicate::str::contains("No SQL statements"));
}

#[test]
fn test_aggregate_query() {
    let mut csv_file = NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(csv_file, "category,amount").unwrap();
    writeln!(csv_file, "A,10").unwrap();
    writeln!(csv_file, "A,20").unwrap();
    writeln!(csv_file, "B,30").unwrap();
    writeln!(csv_file, "B,40").unwrap();

    // Use explicit table name
    qlm_cmd()
        .env("DOUBLEWORD_API_KEY", "test-key")
        .arg("-t")
        .arg(format!("data={}", csv_file.path().display()))
        .arg("-c")
        .arg("SELECT category, SUM(amount) as total FROM data GROUP BY category ORDER BY category;")
        .assert()
        .success();
}
