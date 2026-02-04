use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("Batch failed: {0}")]
    BatchFailed(String),
    #[error("Batch expired")]
    BatchExpired,
    #[error("Batch cancelled")]
    BatchCancelled,
    #[error("Missing output file")]
    MissingOutputFile,
    #[error("Result not found for request: {0}")]
    ResultNotFound(String),
}

#[derive(Debug, Clone)]
pub struct EmbeddingClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    dimensions: usize,
}

// Batch request for embeddings
#[derive(Serialize)]
struct BatchRequest {
    custom_id: String,
    method: String,
    url: String,
    body: EmbeddingRequestBody,
}

#[derive(Serialize)]
struct EmbeddingRequestBody {
    model: String,
    input: String,
}

// Response types
#[derive(Deserialize, Debug)]
struct FileUploadResponse {
    id: String,
}

#[derive(Deserialize, Debug)]
struct BatchResponse {
    id: String,
    status: String,
    output_file_id: Option<String>,
    request_counts: Option<RequestCounts>,
}

#[derive(Deserialize, Debug)]
struct RequestCounts {
    total: u64,
    completed: u64,
    failed: u64,
}

#[derive(Deserialize, Debug)]
struct BatchResultLine {
    custom_id: String,
    response: Option<BatchResultResponse>,
    error: Option<BatchResultError>,
}

#[derive(Deserialize, Debug)]
struct BatchResultResponse {
    body: EmbeddingResponse,
}

#[derive(Deserialize, Debug)]
struct BatchResultError {
    message: String,
}

#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            dimensions,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn check_response(resp: reqwest::Response) -> Result<reqwest::Response, EmbeddingError> {
        if resp.status().is_success() {
            Ok(resp)
        } else {
            let status = resp.status().as_u16();
            let message = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(EmbeddingError::Api { status, message })
        }
    }

    /// Embed multiple texts using the batch API
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_inner(texts, true).await
    }

    /// Embed multiple texts without progress bars
    pub async fn embed_batch_quiet(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_inner(texts, false).await
    }

    async fn embed_batch_inner(
        &self,
        texts: Vec<String>,
        show_progress: bool,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let count = texts.len();

        // Build JSONL content
        let jsonl = self.build_jsonl(&texts)?;
        let upload_size = jsonl.len() as u64;

        // Upload progress
        let upload_progress = if show_progress {
            let pb = ProgressBar::new(upload_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.cyan} Uploading embeddings [{bar:30.cyan/dim}] {bytes}/{total_bytes} ({msg})")
                    .unwrap()
                    .progress_chars("█▓▒░"),
            );
            pb.set_message(format!("{} texts", count));
            pb.enable_steady_tick(Duration::from_millis(80));
            Some(pb)
        } else {
            None
        };

        let file_id = self.upload_file(&jsonl).await?;

        if let Some(ref pb) = upload_progress {
            pb.set_position(upload_size);
            pb.finish_and_clear();
        }

        // Create batch
        let batch_id = self.create_batch(&file_id).await?;

        // Poll and download results
        let results = if show_progress {
            self.poll_and_download(&batch_id, count).await?
        } else {
            self.poll_and_download_quiet(&batch_id, count).await?
        };

        Ok(results)
    }

    fn build_jsonl(&self, texts: &[String]) -> Result<String, EmbeddingError> {
        let mut lines = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let batch_req = BatchRequest {
                custom_id: format!("emb-{}", i),
                method: "POST".to_string(),
                url: "/v1/embeddings".to_string(),
                body: EmbeddingRequestBody {
                    model: self.model.clone(),
                    input: text.clone(),
                },
            };

            lines.push(serde_json::to_string(&batch_req)?);
        }

        Ok(lines.join("\n"))
    }

    async fn upload_file(&self, jsonl: &str) -> Result<String, EmbeddingError> {
        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part(
                "file",
                reqwest::multipart::Part::text(jsonl.to_string())
                    .file_name("embedding_batch.jsonl")
                    .mime_str("application/jsonl")
                    .unwrap(),
            );

        let resp = self
            .client
            .post(format!("{}/files", self.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                // Print full error chain for debugging
                eprintln!("Upload error: {}", e);
                let err_ref: &dyn StdError = &e;
                let mut source = err_ref.source();
                while let Some(s) = source {
                    eprintln!("  caused by: {}", s);
                    source = s.source();
                }
                EmbeddingError::Http(e)
            })?;

        let resp = Self::check_response(resp).await?;
        let file_resp: FileUploadResponse = resp.json().await?;

        Ok(file_resp.id)
    }

    async fn create_batch(&self, input_file_id: &str) -> Result<String, EmbeddingError> {
        let body = serde_json::json!({
            "input_file_id": input_file_id,
            "endpoint": "/v1/embeddings",
            "completion_window": "24h"
        });

        let resp = self
            .client
            .post(format!("{}/batches", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let resp = Self::check_response(resp).await?;
        let batch_resp: BatchResponse = resp.json().await?;

        Ok(batch_resp.id)
    }

    async fn poll_and_download(
        &self,
        batch_id: &str,
        expected_count: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let multi = MultiProgress::new();

        let process_progress = multi.add(ProgressBar::new(expected_count as u64));
        process_progress.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} Embedding [{bar:30.cyan/dim}] {pos}/{len} texts ({msg})",
                )
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        process_progress.enable_steady_tick(Duration::from_millis(80));

        let download_progress = multi.add(ProgressBar::new(expected_count as u64));
        download_progress.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} Downloaded [{bar:30.cyan/dim}] {pos}/{len} embeddings ({bytes})",
                )
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        download_progress.enable_steady_tick(Duration::from_millis(80));

        let mut results_map: HashMap<String, Vec<f32>> = HashMap::new();
        let mut output_file_id: Option<String> = None;
        let mut download_offset: u64 = 0;
        let mut bytes_downloaded: u64 = 0;
        let mut batch_complete = false;

        loop {
            let resp = self
                .client
                .get(format!("{}/batches/{}", self.base_url, batch_id))
                .bearer_auth(&self.api_key)
                .send()
                .await?;

            let resp = Self::check_response(resp).await?;
            let batch_resp: BatchResponse = resp.json().await?;

            if let Some(ref counts) = batch_resp.request_counts {
                process_progress.set_length(counts.total);
                process_progress.set_position(counts.completed + counts.failed);
            }

            if output_file_id.is_none() {
                output_file_id = batch_resp.output_file_id.clone();
            }

            match batch_resp.status.as_str() {
                "completed" => {
                    if let Some(ref counts) = batch_resp.request_counts {
                        process_progress.set_position(counts.total);
                    }
                    process_progress.finish_with_message("✓ complete");
                    batch_complete = true;
                }
                "failed" => {
                    process_progress.finish_with_message("✗ failed");
                    download_progress.finish_with_message("✗ aborted");
                    return Err(EmbeddingError::BatchFailed("Batch processing failed".to_string()));
                }
                "expired" => {
                    process_progress.finish_with_message("✗ expired");
                    download_progress.finish_with_message("✗ aborted");
                    return Err(EmbeddingError::BatchExpired);
                }
                "cancelled" => {
                    process_progress.finish_with_message("✗ cancelled");
                    download_progress.finish_with_message("✗ aborted");
                    return Err(EmbeddingError::BatchCancelled);
                }
                status => {
                    process_progress.set_message(status.to_string());
                }
            }

            if let Some(ref file_id) = output_file_id {
                let (new_results, new_offset, bytes, is_complete) =
                    self.download_partial(file_id, download_offset).await?;

                bytes_downloaded += bytes;

                for line in new_results.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(result) = serde_json::from_str::<BatchResultLine>(line) {
                        let embedding = if let Some(resp) = result.response {
                            resp.body
                                .data
                                .into_iter()
                                .next()
                                .map(|d| d.embedding)
                                .unwrap_or_default()
                        } else if let Some(err) = result.error {
                            eprintln!("Embedding error for {}: {}", result.custom_id, err.message);
                            vec![0.0; self.dimensions]
                        } else {
                            vec![0.0; self.dimensions]
                        };
                        results_map.insert(result.custom_id, embedding);
                    }
                }

                let downloaded = results_map.len() as u64;
                let processed = process_progress.position();
                download_progress.set_position(downloaded.min(processed));
                download_progress.set_message(format!("{} bytes", bytes_downloaded));

                download_offset = new_offset;

                if batch_complete && is_complete {
                    download_progress.set_position(downloaded);
                    download_progress.finish_with_message(format!("✓ {} bytes", bytes_downloaded));
                    break;
                }
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Reconstruct results in original order
        let mut results = Vec::with_capacity(expected_count);
        for i in 0..expected_count {
            let custom_id = format!("emb-{}", i);
            let result = results_map
                .remove(&custom_id)
                .ok_or_else(|| EmbeddingError::ResultNotFound(custom_id))?;
            results.push(result);
        }

        Ok(results)
    }

    async fn poll_and_download_quiet(
        &self,
        batch_id: &str,
        expected_count: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results_map: HashMap<String, Vec<f32>> = HashMap::new();
        let mut output_file_id: Option<String> = None;
        let mut download_offset: u64 = 0;
        let mut batch_complete = false;

        loop {
            let resp = self
                .client
                .get(format!("{}/batches/{}", self.base_url, batch_id))
                .bearer_auth(&self.api_key)
                .send()
                .await?;

            let resp = Self::check_response(resp).await?;
            let batch_resp: BatchResponse = resp.json().await?;

            if output_file_id.is_none() {
                output_file_id = batch_resp.output_file_id.clone();
            }

            match batch_resp.status.as_str() {
                "completed" => {
                    batch_complete = true;
                }
                "failed" => {
                    return Err(EmbeddingError::BatchFailed("Batch processing failed".to_string()));
                }
                "expired" => return Err(EmbeddingError::BatchExpired),
                "cancelled" => return Err(EmbeddingError::BatchCancelled),
                _ => {}
            }

            if let Some(ref file_id) = output_file_id {
                let (new_results, new_offset, _, is_complete) =
                    self.download_partial(file_id, download_offset).await?;

                for line in new_results.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(result) = serde_json::from_str::<BatchResultLine>(line) {
                        let embedding = if let Some(resp) = result.response {
                            resp.body
                                .data
                                .into_iter()
                                .next()
                                .map(|d| d.embedding)
                                .unwrap_or_default()
                        } else if let Some(err) = result.error {
                            eprintln!("Embedding error for {}: {}", result.custom_id, err.message);
                            vec![0.0; self.dimensions]
                        } else {
                            vec![0.0; self.dimensions]
                        };
                        results_map.insert(result.custom_id, embedding);
                    }
                }

                download_offset = new_offset;

                if batch_complete && is_complete {
                    break;
                }
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        let mut results = Vec::with_capacity(expected_count);
        for i in 0..expected_count {
            let custom_id = format!("emb-{}", i);
            let result = results_map
                .remove(&custom_id)
                .ok_or_else(|| EmbeddingError::ResultNotFound(custom_id))?;
            results.push(result);
        }

        Ok(results)
    }

    async fn download_partial(
        &self,
        file_id: &str,
        offset: u64,
    ) -> Result<(String, u64, u64, bool), EmbeddingError> {
        let url = if offset > 0 {
            format!(
                "{}/files/{}/content?offset={}",
                self.base_url, file_id, offset
            )
        } else {
            format!("{}/files/{}/content", self.base_url, file_id)
        };

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let resp = Self::check_response(resp).await?;

        let is_incomplete = resp
            .headers()
            .get("X-Incomplete")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == "true")
            .unwrap_or(false);

        let last_line = resp
            .headers()
            .get("X-Last-Line")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(offset);

        let mut content = Vec::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            content.extend_from_slice(&chunk);
        }

        let bytes_read = content.len() as u64;
        let content_str = String::from_utf8_lossy(&content).to_string();

        let new_offset = if last_line > offset {
            last_line
        } else {
            offset + bytes_read
        };

        Ok((content_str, new_offset, bytes_read, !is_incomplete))
    }
}
