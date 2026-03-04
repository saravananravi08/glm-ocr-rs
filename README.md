# glm-ocr

A pure Rust OCR inference engine powered by the [GLM-OCR](https://huggingface.co/unsloth/GLM-OCR) vision-language model. Built on the [candle](https://github.com/huggingface/candle) ML framework — no Python or PyTorch required.

## Features

- **Single binary** — no Python, no pip, no virtual environments
- **CPU and GPU** — runs on any x86 CPU; optional CUDA support for ~10x speedup
- **Q8_0 / Q4_0 quantization** — faster CPU inference with minimal quality loss
- **Layout detection** — automatic document segmentation using PP-DocLayout-M
- **Structured output** — JSON output with typed sections, bounding boxes, key-value pairs, and parsed tables
- **Large document support** — automatic strip-based processing for high-resolution pages

## Quick Start

### Prerequisites

- Rust 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- For GPU: CUDA toolkit and an NVIDIA GPU
- For layout detection: ONNX Runtime (`libonnxruntime.so` — see [Layout Detection](#layout-detection))

### Build

```bash
# CPU only (default)
cargo build --release

# With CUDA GPU support
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda
```

> Set `CUDA_COMPUTE_CAP` to match your GPU architecture (e.g., `86` for RTX 3060/4060, `89` for RTX 4090). Check yours with `nvidia-smi --query-gpu=compute_cap --format=csv`.

### Run

```bash
# Basic text recognition
./target/release/glm-ocr --image document.png

# With Q8_0 quantization (recommended for CPU)
./target/release/glm-ocr --image document.png --quantize q8_0

# With layout detection
./target/release/glm-ocr --image document.png --quantize q8_0 --layout

# JSON structured output
./target/release/glm-ocr --image document.png --quantize q8_0 --layout --json

# GPU inference
./target/release/glm-ocr --image document.png --gpu 0

# GPU + layout + JSON
./target/release/glm-ocr --image document.png --gpu 0 --layout --json
```

The model weights (~2.65 GB) are automatically downloaded from HuggingFace on first run.

## CLI Options

| Flag | Description |
|------|-------------|
| `--image <PATH>` | Path to the input image (required) |
| `--prompt <TEXT>` | OCR prompt (default: `"Text Recognition:"`) |
| `--max-tokens <N>` | Maximum tokens to generate (default: 8192) |
| `--quantize <LEVEL>` | Quantize text decoder: `q8_0` (recommended) or `q4_0` |
| `--layout` | Enable layout detection for document segmentation |
| `--json` | Output structured JSON (requires `--layout`) |
| `--gpu <N>` | Use CUDA GPU N for inference (e.g., `--gpu 0`) |
| `--model-id <ID>` | Custom HuggingFace model ID (default: `unsloth/GLM-OCR`) |

## Supported Prompts

GLM-OCR supports three built-in prompts:

| Prompt | Use Case |
|--------|----------|
| `Text Recognition:` | General text and document OCR (default) |
| `Table Recognition:` | Outputs HTML table structure |
| `Formula Recognition:` | LaTeX math formula recognition |

In `--layout` mode, the prompt is automatically selected per region type.

## Layout Detection

Layout mode uses [PP-DocLayout-M](https://github.com/PaddlePaddle/PaddleOCR) (PicoDet-L) to segment the document page into regions before running OCR on each one. This improves accuracy on complex multi-region documents (invoices, purchase orders, reports).

### Setup

1. Download the ONNX model:
   ```bash
   # From PaddleOCR model zoo or convert from PaddlePaddle format
   # Place the model file as pp-doclayout-m.onnx
   ```

2. Set the model path:
   ```bash
   export LAYOUT_MODEL_PATH=/path/to/pp-doclayout-m.onnx
   ```

3. Ensure ONNX Runtime is available:
   ```bash
   export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
   ```

### Detected Region Types

The layout detector recognizes 23 region types including: `text`, `table`, `image`, `doc_title`, `paragraph_title`, `header`, `footer`, `formula`, `seal`, `chart`, and more.

## Structured JSON Output

With `--layout --json`, the output is a structured document:

```json
{
  "width": 1240,
  "height": 1754,
  "sections": [
    {
      "label": "text",
      "bbox": [50.0, 100.0, 1190.0, 400.0],
      "text": "Order Date: 16/03/2020\nPO Number: 4500012345",
      "key_values": [
        { "key": "Order Date", "value": "16/03/2020" },
        { "key": "PO Number", "value": "4500012345" }
      ],
      "table": null
    },
    {
      "label": "table",
      "bbox": [50.0, 420.0, 1190.0, 900.0],
      "text": "Item Description Qty Amount\n1 Widget A 10 500.00",
      "key_values": [],
      "table": {
        "headers": ["Item", "Description", "Qty", "Amount"],
        "rows": [["1", "Widget A", "10", "500.00"]]
      }
    }
  ]
}
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
glm-ocr = { path = "path/to/glm-ocr" }
```

```rust
use glm_ocr::GlmOcr;

fn main() -> anyhow::Result<()> {
    let image = image::open("document.png")?;

    // Basic OCR
    let ocr = GlmOcr::new(None, Some("q8_0"))?;
    let text = ocr.recognize(&image, "Text Recognition:")?;
    println!("{text}");

    // Layout-aware structured OCR
    let mut layout = glm_ocr::layout::LayoutDetector::new()?;
    let doc = ocr.recognize_layout_structured(&image, &mut layout, 8192)?;

    // Output as markdown
    println!("{}", doc.to_markdown());

    // Output as JSON
    println!("{}", doc.to_json()?);

    // Access structured data
    for section in &doc.sections {
        println!("[{}] {}", section.label, section.bbox[0]);
        for kv in &section.key_values {
            println!("  {} = {}", kv.key, kv.value);
        }
        if let Some(table) = &section.table {
            println!("  Table: {} cols x {} rows", table.headers.len(), table.rows.len());
        }
    }

    Ok(())
}
```

### GPU Usage

```rust
use candle_core::Device;
use glm_ocr::GlmOcr;

let device = Device::new_cuda(0)?;
let ocr = GlmOcr::new_with_device(None, None, device)?; // quantization auto-skipped on GPU
```

## HTTP API Server

A built-in Axum server with hybrid GPU/CPU worker routing. Models stay loaded in memory — no per-request loading overhead. If compiled with CUDA and a GPU is available, the server automatically loads both a GPU and a CPU worker for concurrent request handling.

### Build & Run

```bash
# CPU only
cargo build --release --bin glm-ocr-server

# With GPU
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda --bin glm-ocr-server

# Start (GPU auto-detected at runtime)
LAYOUT_MODEL_PATH=./pp-doclayout-m.onnx ./target/release/glm-ocr-server --quantize q8_0 --port 8080
```

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Listen port |
| `--quantize` | none | `q8_0` or `q4_0` (CPU only, auto-skipped on GPU) |
| `--model-id` | `unsloth/GLM-OCR` | HuggingFace model ID |
| `--max-tokens` | `8192` | Default max tokens per region |

### API Reference

#### `GET /health`

Returns server status and worker availability.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "workers": [
    { "device": "gpu:0", "busy": false },
    { "device": "cpu", "busy": false }
  ]
}
```

#### `POST /ocr`

Accepts a multipart form with an image and returns OCR results.

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | required | Image file (PNG, JPEG, etc.) |
| `prompt` | string | `Text Recognition:` | OCR prompt |
| `layout` | bool | `false` | Enable layout detection |
| `json` | bool | `false` | Return structured document (requires `layout=true`) |
| `max_tokens` | int | `8192` | Max tokens per region |
| `device` | string | `auto` | Worker routing: `auto`, `gpu`, or `cpu` |

**Basic text recognition:**

```bash
curl -X POST http://localhost:8080/ocr -F image=@document.png
```

```json
{
  "text": "Invoice No: INV-2024-001\nDate: 15/03/2024\n...",
  "device": "gpu:0",
  "elapsed_ms": 1234
}
```

**Layout detection (markdown):**

```bash
curl -X POST http://localhost:8080/ocr \
  -F image=@document.png \
  -F layout=true
```

```json
{
  "text": "## Invoice\n\n### Items\n\n| Item | Qty | Amount |\n| --- | --- | --- |\n| Widget A | 10 | 500.00 |\n\n_Page 1 of 1_",
  "device": "gpu:0",
  "elapsed_ms": 9300
}
```

**Layout detection (structured JSON):**

```bash
curl -X POST http://localhost:8080/ocr \
  -F image=@document.png \
  -F layout=true \
  -F json=true
```

```json
{
  "document": {
    "width": 1240,
    "height": 1754,
    "sections": [
      {
        "label": "doc_title",
        "bbox": [100.0, 50.0, 400.0, 80.0],
        "text": "Invoice",
        "key_values": [],
        "table": null
      },
      {
        "label": "text",
        "bbox": [50.0, 100.0, 600.0, 300.0],
        "text": "Invoice No: INV-2024-001\nDate: 15/03/2024",
        "key_values": [
          { "key": "Invoice No", "value": "INV-2024-001" },
          { "key": "Date", "value": "15/03/2024" }
        ],
        "table": null
      },
      {
        "label": "table",
        "bbox": [50.0, 320.0, 600.0, 500.0],
        "text": "Item Qty Amount\nWidget A 10 500.00",
        "key_values": [],
        "table": {
          "headers": ["Item", "Qty", "Amount"],
          "rows": [["Widget A", "10", "500.00"]]
        }
      }
    ]
  },
  "device": "gpu:0",
  "elapsed_ms": 9300
}
```

**Force a specific device:**

```bash
# Force CPU (useful when GPU is busy)
curl -X POST http://localhost:8080/ocr -F image=@document.png -F device=cpu

# Force GPU (returns 503 if no GPU worker)
curl -X POST http://localhost:8080/ocr -F image=@document.png -F device=gpu
```

**Error responses:**

```json
{"error": "missing 'image' field"}
{"error": "json=true requires layout=true"}
{"error": "no worker available for requested device"}
```

### Integration Examples

**Python:**

```python
import requests

# Basic OCR
resp = requests.post("http://localhost:8080/ocr",
    files={"image": open("document.png", "rb")})
print(resp.json()["text"])

# Layout + structured JSON
resp = requests.post("http://localhost:8080/ocr",
    files={"image": open("document.png", "rb")},
    data={"layout": "true", "json": "true"})
doc = resp.json()["document"]
for section in doc["sections"]:
    print(f"[{section['label']}] {section['text'][:80]}")
```

**JavaScript (fetch):**

```javascript
const form = new FormData();
form.append("image", fs.readFileSync("document.png"), "document.png");
form.append("layout", "true");
form.append("json", "true");

const res = await fetch("http://localhost:8080/ocr", { method: "POST", body: form });
const { document, elapsed_ms } = await res.json();
console.log(`${document.sections.length} sections in ${elapsed_ms}ms`);
```

## Performance

Benchmarks on a single document page (A4, 150 DPI):

| Mode | Time per Page | Notes |
|------|--------------|-------|
| CPU (F32) | ~300s | No quantization |
| CPU (Q8_0) | ~78s | Recommended for CPU |
| GPU (RTX 4060, F16) | ~9s | Native F16, no quantization |

> Quantization is automatically skipped on GPU — native F16 matmul is faster than QMatMul's F32 dequantize path. Inference speed is bottlenecked by autoregressive token generation (each token depends on the previous).

## Architecture

```
GLM-OCR (0.9B parameters)
├── CogViT Vision Encoder (24 layers, 1024 hidden, 16 heads)
│   ├── Patch Embedding (14x14 patches, merge_size=2)
│   ├── Vision Rotary Embeddings (head_dim/2 = 32)
│   └── Spatial Downsample (Conv2d merger)
├── Text Decoder (16 layers, 1536 hidden, 16 attn heads, 8 KV heads)
│   ├── Grouped-Query Attention (GQA)
│   ├── Multi-axis Rotary Position Embeddings (mRoPE: [16,24,24])
│   └── 4-norm blocks (input→attn→post_self_attn→post_attn→mlp→post_mlp)
└── Layout Detector (PP-DocLayout-M, PicoDet-L ONNX)
    └── 23 region classes, 640x640 input
```

## License

MIT
