# glm-ocr-rs

> *The lightest VLM that still gets document structure.*

Pure Rust implementation of [GLM-OCR](https://huggingface.co/unsloth/GLM-OCR) — a 0.9B vision-language model for document understanding. Built from scratch on [candle](https://github.com/huggingface/candle), every layer hand-ported to Rust. The entire model runs from a single binary.

## Why GLM-OCR?

At 0.9B parameters (~2.5 GB quantized), GLM-OCR is small enough to run on a laptop GPU but smart enough to understand tables, formulas, and complex document layouts — not just read text line by line.

## What You Get

- **One binary. Zero dependencies.** Ship it, `scp` it, Docker it — it just works
- **CPU or GPU** — Q8_0 quantized CPU or CUDA F16 for when you're impatient
- **Layout-aware OCR** — doesn't just dump text, it *understands* your document structure
- **Structured JSON output** — sections, bounding boxes, key-value pairs, parsed tables — ready for your pipeline

## Quick Start

```bash
# Build it
cargo build --release

# Run it
./target/release/glm-ocr --image invoice.png --quantize

# Layout detection + tables + JSON
./target/release/glm-ocr --image invoice.png --quantize --layout --json

# Got a GPU? (~10x faster)
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda
./target/release/glm-ocr --image invoice.png --gpu 0 --layout
```

First run downloads the model (~2.65 GB) from HuggingFace. After that, it's all local.

## What It Actually Does

Feed it a document image. It gives you back structured, readable text.

**Without layout** — straightforward text recognition:
```bash
./target/release/glm-ocr --image page.png --quantize
```

**With layout** — detects regions (titles, tables, text blocks, headers, footers), OCRs each one with the right prompt, assembles in reading order:
```bash
./target/release/glm-ocr --image page.png --quantize --layout
```

**With JSON** — machine-readable structured output:
```bash
./target/release/glm-ocr --image page.png --quantize --layout --json
```

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
      "text": "...",
      "key_values": [],
      "table": {
        "headers": ["Item", "Description", "Qty", "Amount"],
        "rows": [["1", "Widget A", "10", "500.00"]]
      }
    }
  ]
}
```

## CLI Reference

| Flag | What it does |
|------|-------------|
| `--image <PATH>` | Input image (required) |
| `--prompt <TEXT>` | OCR prompt — `"Text Recognition:"`, `"Table Recognition:"`, or `"Formula Recognition:"` |
| `--quantize` | Q8_0 quantization — 2-3x faster on CPU, basically free |
| `--layout` | Smart document segmentation via PP-DocLayout-M |
| `--json` | Structured JSON output (needs `--layout`) |
| `--gpu <N>` | CUDA GPU inference (build with `--features cuda`) |
| `--max-tokens <N>` | Token limit per region (default: 8192) |
| `--model-id <ID>` | Custom HuggingFace model (default: `unsloth/GLM-OCR`) |

## Use It As a Library

```rust
use glm_ocr::GlmOcr;

let image = image::open("document.png")?;

// Simple OCR
let ocr = GlmOcr::new(None, true)?;
let text = ocr.recognize(&image, "Text Recognition:")?;

// Layout-aware structured OCR
let mut layout = glm_ocr::layout::LayoutDetector::new()?;
let doc = ocr.recognize_layout_structured(&image, &mut layout, 8192)?;

// Markdown or JSON — your choice
println!("{}", doc.to_markdown());
println!("{}", doc.to_json()?);

// Or dig into the structure
for section in &doc.sections {
    println!("[{}] @ {:?}", section.label, section.bbox);
    for kv in &section.key_values {
        println!("  {} = {}", kv.key, kv.value);
    }
    if let Some(table) = &section.table {
        println!("  {}x{} table", table.headers.len(), table.rows.len());
    }
}
```

GPU:
```rust
let device = candle_core::Device::new_cuda(0)?;
let ocr = GlmOcr::new_with_device(None, false, device)?;
```

## Performance

| Mode | Time / Page | Vibe |
|------|------------|------|
| CPU | ~300s | Go make coffee |
| CPU + Q8_0 | ~65-95s | Scroll Twitter once |
| GPU (RTX 4060) | ~12s | Blink and it's done |

VLMs generate tokens one at a time — each token waits for the previous one. That's physics, not a bug. GPU helps because it crunches each token's matrix math faster, not because it parallelizes generation.

## Layout Detection Setup

Uses [PP-DocLayout-M](https://github.com/PaddlePaddle/PaddleOCR) to segment pages into 23 region types — text, table, title, header, footer, formula, image, seal, chart, and more.

```bash
export LAYOUT_MODEL_PATH=/path/to/pp-doclayout-m.onnx
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

## Under the Hood

```
GLM-OCR (0.9B params)
├── CogViT Vision Encoder
│   ├── 24 transformer layers (1024 hidden, 16 heads)
│   ├── 14x14 patch embedding with 2x2 merge
│   ├── Vision rotary embeddings (head_dim/2)
│   └── Conv2d spatial downsampler
├── GLM Text Decoder
│   ├── 16 transformer layers (1536 hidden)
│   ├── Grouped-Query Attention (16 attn / 8 KV heads)
│   ├── Multi-axis RoPE [16, 24, 24]
│   └── 4-norm residual blocks
└── PP-DocLayout-M (PicoDet-L ONNX)
    └── 23-class document layout detection
```

Every one of these components — hand-ported to Rust. No bindings. No FFI wrappers. Pure, native Rust all the way down to the matrix multiplications.

## Build Options

```bash
# CPU (default)
cargo build --release

# GPU — set CUDA_COMPUTE_CAP for your card
# RTX 3060/4060: 86 | RTX 4090: 89 | A100: 80
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda
```

### Prerequisites
- Rust 1.75+
- For GPU: CUDA toolkit + NVIDIA GPU
- For layout: ONNX Runtime shared library

## License

MIT
