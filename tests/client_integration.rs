//! Integration tests for the LLM client using mock servers
//!
//! These tests verify the client correctly communicates with the API
//! without making real API calls.

use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Helper to create a mock batch response
fn mock_batch_response(id: &str, status: &str, output_file_id: Option<&str>) -> serde_json::Value {
    let mut response = serde_json::json!({
        "id": id,
        "status": status,
        "request_counts": {
            "total": 1,
            "completed": if status == "completed" { 1 } else { 0 },
            "failed": 0
        }
    });

    if let Some(file_id) = output_file_id {
        response["output_file_id"] = serde_json::json!(file_id);
    }

    response
}

/// Helper to create a mock result line
fn mock_result_line(custom_id: &str, content: &str) -> String {
    serde_json::json!({
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": content
                    }
                }]
            }
        }
    })
    .to_string()
}

#[tokio::test]
async fn test_process_prompts_empty() {
    // Empty prompts should return immediately without API calls
    let client = qlm::client::LlmClient::new("http://unused", "key", "model");
    let result = client.process_prompts(vec![]).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_api_error_unauthorized() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Invalid API key"))
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "bad-key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("401"));
}

#[tokio::test]
async fn test_api_error_rate_limit() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(429).set_body_string("Rate limit exceeded"))
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("429"));
}

#[tokio::test]
async fn test_batch_failed_status() {
    let mock_server = MockServer::start().await;

    // Mock file upload
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": "file-123"})),
        )
        .mount(&mock_server)
        .await;

    // Mock batch creation
    Mock::given(method("POST"))
        .and(path("/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_batch_response(
            "batch-456",
            "validating",
            None,
        )))
        .mount(&mock_server)
        .await;

    // Mock batch status - failed
    Mock::given(method("GET"))
        .and(path_regex(r"/batches/batch-456"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_batch_response("batch-456", "failed", None)),
        )
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("failed"));
}

#[tokio::test]
async fn test_batch_expired_status() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": "file-123"})),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_batch_response(
            "batch-456",
            "validating",
            None,
        )))
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"/batches/batch-456"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_batch_response("batch-456", "expired", None)),
        )
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expired"));
}

#[tokio::test]
async fn test_batch_cancelled_status() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": "file-123"})),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_batch_response(
            "batch-456",
            "validating",
            None,
        )))
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"/batches/batch-456"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_batch_response("batch-456", "cancelled", None)),
        )
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cancelled"));
}

#[tokio::test]
async fn test_server_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let result = client.process_prompts(vec!["test".to_string()]).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("500"));
}

#[tokio::test]
async fn test_file_upload_creates_correct_request() {
    let mock_server = MockServer::start().await;

    let file_mock = Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": "file-test"})),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // We don't complete the full flow, just verify the file upload is attempted
    Mock::given(method("POST"))
        .and(path("/batches"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_batch_response("batch-1", "failed", None)),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"/batches/.*"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_batch_response("batch-1", "failed", None)),
        )
        .mount(&mock_server)
        .await;

    let client = qlm::client::LlmClient::new(mock_server.uri(), "key", "model");
    let _ = client.process_prompts(vec!["test prompt".to_string()]).await;

    // Verify file upload was called
    // (the expect(1) on file_mock will verify this)
}
