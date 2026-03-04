use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use glm_ocr::layout::LayoutDetector;
use glm_ocr::GlmOcr;

#[derive(Parser)]
#[command(name = "glm-ocr")]
#[command(about = "Pure Rust GLM-OCR inference engine")]
struct Cli {
    /// Path to the image file
    #[arg(short, long)]
    image: PathBuf,

    /// OCR prompt (default: "Text Recognition:")
    #[arg(short, long, default_value = "Text Recognition:")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(short, long, default_value_t = 8192)]
    max_tokens: usize,

    /// HuggingFace model ID (default: unsloth/GLM-OCR)
    #[arg(long)]
    model_id: Option<String>,

    /// Use layout detection for intelligent document segmentation
    #[arg(long)]
    layout: bool,

    /// Quantize text decoder weights for faster inference. Levels: q8_0 (default), q4_0 (faster, lower quality)
    #[arg(long)]
    quantize: Option<String>,

    /// Output structured JSON instead of markdown (requires --layout)
    #[arg(long)]
    json: bool,

    /// Use CUDA GPU for inference (requires --features cuda). Specify GPU index, e.g. --gpu 0
    #[arg(long)]
    gpu: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Load image
    let image = image::open(&cli.image)?;

    // Select device
    let device = match cli.gpu {
        Some(gpu_id) => {
            let dev = Device::new_cuda(gpu_id)?;
            eprintln!("Using CUDA GPU {gpu_id}");
            dev
        }
        None => Device::Cpu,
    };

    // Initialize OCR model
    let start = Instant::now();
    let model_id = cli.model_id.as_deref();
    let quantize = cli.quantize.as_deref();
    let ocr = GlmOcr::new_with_device(model_id, quantize, device)?;
    let load_time = start.elapsed();
    eprintln!("OCR model loaded in {load_time:.2?}");

    // Run OCR
    let start = Instant::now();
    if cli.layout {
        eprintln!("Loading layout detection model...");
        let layout_start = Instant::now();
        let mut layout = LayoutDetector::new()?;
        eprintln!("Layout model loaded in {:.2?}", layout_start.elapsed());

        if cli.json {
            let doc = ocr.recognize_layout_structured(&image, &mut layout, cli.max_tokens)?;
            let json = doc.to_json()?;
            println!("{json}");
        } else {
            let result = ocr.recognize_with_layout(&image, &mut layout, cli.max_tokens)?;
            println!("{result}");
        }
    } else {
        if cli.json {
            anyhow::bail!("--json requires --layout");
        }
        let result = ocr.recognize_with_max_tokens(&image, &cli.prompt, cli.max_tokens)?;
        println!("{result}");
    };
    let inference_time = start.elapsed();
    eprintln!("Inference completed in {inference_time:.2?}");

    Ok(())
}
