use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use clap::Parser;
use serde::Serialize;
use tokio::sync::Semaphore;

use glm_ocr::layout::LayoutDetector;
use glm_ocr::GlmOcr;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "glm-ocr-server")]
#[command(about = "GLM-OCR HTTP API server")]
struct Cli {
    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// HuggingFace model ID (default: unsloth/GLM-OCR)
    #[arg(long)]
    model_id: Option<String>,

    /// Quantization level: q8_0, q4_0
    #[arg(long)]
    quantize: Option<String>,

    /// Maximum tokens per region/request
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
}

// ---------------------------------------------------------------------------
// Worker & State
// ---------------------------------------------------------------------------

struct Worker {
    ocr: Mutex<GlmOcr>,
    layout: Mutex<LayoutDetector>,
    semaphore: Arc<Semaphore>,
    device_name: String,
}

struct AppState {
    workers: Vec<Arc<Worker>>,
    default_max_tokens: usize,
}

#[derive(Clone, Copy)]
enum DevicePref {
    Auto,
    Gpu,
    Cpu,
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    workers: Vec<WorkerStatus>,
}

#[derive(Serialize)]
struct WorkerStatus {
    device: String,
    busy: bool,
}

#[derive(Serialize)]
struct OcrResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<glm_ocr::DocumentLayout>,
    device: String,
    elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

enum AppError {
    BadRequest(String),
    NoWorker,
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::NoWorker => (
                StatusCode::SERVICE_UNAVAILABLE,
                "no worker available for requested device".into(),
            ),
            AppError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        let body = serde_json::json!({ "error": msg });
        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Worker selection
// ---------------------------------------------------------------------------

impl AppState {
    async fn acquire_worker(
        &self,
        pref: DevicePref,
    ) -> Result<(Arc<Worker>, tokio::sync::OwnedSemaphorePermit), AppError> {
        let is_gpu = |w: &Worker| w.device_name.starts_with("gpu");

        let candidates: Vec<&Arc<Worker>> = self
            .workers
            .iter()
            .filter(|w| match pref {
                DevicePref::Gpu => is_gpu(w),
                DevicePref::Cpu => !is_gpu(w),
                DevicePref::Auto => true,
            })
            .collect();

        if candidates.is_empty() {
            return Err(AppError::NoWorker);
        }

        // Order: GPU first when auto
        let ordered: Vec<Arc<Worker>> = match pref {
            DevicePref::Auto => {
                let mut gpu: Vec<_> = candidates.iter().filter(|w| is_gpu(w)).cloned().cloned().collect();
                let cpu: Vec<_> = candidates.iter().filter(|w| !is_gpu(w)).cloned().cloned().collect();
                gpu.extend(cpu);
                gpu
            }
            _ => candidates.into_iter().cloned().collect(),
        };

        // Try to acquire without waiting (pick first idle worker)
        for w in &ordered {
            if let Ok(permit) = w.semaphore.clone().try_acquire_owned() {
                return Ok((w.clone(), permit));
            }
        }

        // All busy — wait on first candidate
        let w = ordered[0].clone();
        let permit = w
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AppError::NoWorker)?;
        Ok((w, permit))
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let workers = state
        .workers
        .iter()
        .map(|w| WorkerStatus {
            device: w.device_name.clone(),
            busy: w.semaphore.available_permits() == 0,
        })
        .collect();

    Json(HealthResponse {
        status: "ok".into(),
        workers,
    })
}

async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>, AppError> {
    // Parse multipart fields
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut prompt = "Text Recognition:".to_string();
    let mut use_layout = false;
    let mut use_json = false;
    let mut max_tokens = state.default_max_tokens;
    let mut device_pref = DevicePref::Auto;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("multipart error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" => {
                image_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| AppError::BadRequest(format!("failed to read image: {e}")))?
                        .to_vec(),
                );
            }
            "prompt" => {
                prompt = field.text().await.unwrap_or_default();
            }
            "layout" => {
                let val = field.text().await.unwrap_or_default();
                use_layout = val == "true" || val == "1";
            }
            "json" => {
                let val = field.text().await.unwrap_or_default();
                use_json = val == "true" || val == "1";
            }
            "max_tokens" => {
                let val = field.text().await.unwrap_or_default();
                max_tokens = val.parse().unwrap_or(state.default_max_tokens);
            }
            "device" => {
                let val = field.text().await.unwrap_or_default();
                device_pref = match val.as_str() {
                    "gpu" => DevicePref::Gpu,
                    "cpu" => DevicePref::Cpu,
                    _ => DevicePref::Auto,
                };
            }
            _ => {}
        }
    }

    let image_bytes =
        image_bytes.ok_or_else(|| AppError::BadRequest("missing 'image' field".into()))?;

    if use_json && !use_layout {
        return Err(AppError::BadRequest(
            "json=true requires layout=true".into(),
        ));
    }

    // Acquire worker
    let (worker, _permit) = state.acquire_worker(device_pref).await?;
    let device_name = worker.device_name.clone();
    let start = Instant::now();

    // Run inference in blocking thread
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<OcrResult> {
        let image = image::load_from_memory(&image_bytes)
            .map_err(|e| anyhow::anyhow!("invalid image: {e}"))?;

        let ocr = worker
            .ocr
            .lock()
            .map_err(|e| anyhow::anyhow!("ocr lock poisoned: {e}"))?;

        if use_layout {
            let mut layout = worker
                .layout
                .lock()
                .map_err(|e| anyhow::anyhow!("layout lock poisoned: {e}"))?;
            if use_json {
                let doc = ocr.recognize_layout_structured(&image, &mut layout, max_tokens)?;
                Ok(OcrResult::Layout(doc))
            } else {
                let text = ocr.recognize_with_layout(&image, &mut layout, max_tokens)?;
                Ok(OcrResult::Text(text))
            }
        } else {
            let text = ocr.recognize_with_max_tokens(&image, &prompt, max_tokens)?;
            Ok(OcrResult::Text(text))
        }
    })
    .await
    .map_err(|e| AppError::Internal(format!("task join error: {e}")))?
    .map_err(|e| AppError::Internal(format!("{e}")))?;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        OcrResult::Text(text) => Ok(Json(OcrResponse {
            text: Some(text),
            document: None,
            device: device_name,
            elapsed_ms,
        })),
        OcrResult::Layout(doc) => Ok(Json(OcrResponse {
            text: None,
            document: Some(doc),
            device: device_name,
            elapsed_ms,
        })),
    }
}

enum OcrResult {
    Text(String),
    Layout(glm_ocr::DocumentLayout),
}

// ---------------------------------------------------------------------------
// Worker loading
// ---------------------------------------------------------------------------

fn load_worker(cli: &Cli, device: Device, name: &str) -> Result<Worker> {
    let ocr = GlmOcr::new_with_device(cli.model_id.as_deref(), cli.quantize.as_deref(), device)?;
    let layout = LayoutDetector::new()?;
    Ok(Worker {
        ocr: Mutex::new(ocr),
        layout: Mutex::new(layout),
        semaphore: Arc::new(Semaphore::new(1)),
        device_name: name.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let mut workers: Vec<Arc<Worker>> = Vec::new();

    // Try GPU worker (runtime detection, only if compiled with cuda feature)
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                tracing::info!("CUDA GPU detected, loading GPU model...");
                match load_worker(&cli, device, "gpu:0") {
                    Ok(w) => {
                        tracing::info!("GPU worker ready");
                        workers.push(Arc::new(w));
                    }
                    Err(e) => tracing::warn!("Failed to load GPU model: {e}"),
                }
            }
            Err(e) => tracing::info!("CUDA not available: {e}"),
        }
    }

    // Always load CPU worker
    tracing::info!("Loading CPU model...");
    let cpu_worker = load_worker(&cli, Device::Cpu, "cpu")?;
    workers.push(Arc::new(cpu_worker));

    let worker_names: Vec<&str> = workers.iter().map(|w| w.device_name.as_str()).collect();
    tracing::info!("=== GLM-OCR Server ===");
    tracing::info!("Workers: {}", worker_names.join(", "));
    tracing::info!(
        "Quantization: {}",
        cli.quantize.as_deref().unwrap_or("none")
    );
    tracing::info!("Max tokens: {}", cli.max_tokens);

    let state = Arc::new(AppState {
        workers,
        default_max_tokens: cli.max_tokens,
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/ocr", post(ocr_handler))
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("Listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
