pub mod config;
pub mod generation;
pub mod image_processor;
pub mod layout;
pub mod model;
pub mod model_loader;
pub mod nn_utils;
pub mod quantize;
pub mod text;
pub mod tokenizer;
pub mod vision;

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use image::DynamicImage;
use serde::Serialize;

use config::GlmOcrConfig;
use layout::LayoutDetector;
use model::GlmOcrModel;
use model_loader::ModelLoader;
use tokenizer::GlmOcrTokenizer;

/// Structured representation of a full document page after layout OCR.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentLayout {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Sections in reading order (top-to-bottom, left-to-right).
    pub sections: Vec<DocumentSection>,
}

/// A single recognized region within a document.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentSection {
    /// Layout region type (e.g. "text", "table", "doc_title", "footer").
    pub label: String,
    /// Bounding box in image coordinates: [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Raw OCR text from the VLM.
    pub text: String,
    /// Key-value pairs extracted from the text (e.g. "Order Date": "16/03/2020").
    /// Empty for non-text regions or when no key-value patterns are found.
    pub key_values: Vec<KeyValue>,
    /// Parsed table data (only populated for table regions with detectable headers).
    pub table: Option<TableData>,
}

/// A key-value pair extracted from a text region.
#[derive(Debug, Clone, Serialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

/// Structured table data with headers and rows.
#[derive(Debug, Clone, Serialize)]
pub struct TableData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl DocumentLayout {
    /// Render the document as markdown text.
    ///
    /// Produces the same output as `recognize_with_layout()` — section headings,
    /// markdown tables, italic headers, and underlined footers.
    pub fn to_markdown(&self) -> String {
        let formatted: Vec<String> = self
            .sections
            .iter()
            .map(|s| format_region_by_label(&s.text, &s.label))
            .collect();
        deduplicate_adjacent_sections(&formatted)
    }

    /// Serialize the document to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Max total merged patches per strip to keep memory and quality reasonable.
/// 300 merged patches → fast inference, good text quality on CPU.
const MAX_MERGED_PATCHES: u32 = 300;

/// Overlap between adjacent strips in original image pixels.
const STRIP_OVERLAP: u32 = 56;

/// Parse a quantization level string into a GgmlDType.
///
/// Accepts: "q4_0", "q4", "q8_0", "q8", or None for no quantization.
pub fn parse_quantization(s: Option<&str>) -> Result<Option<GgmlDType>> {
    match s {
        None => Ok(None),
        Some(s) => match s.to_lowercase().as_str() {
            "q8_0" | "q8" => Ok(Some(GgmlDType::Q8_0)),
            "q4_0" | "q4" => Ok(Some(GgmlDType::Q4_0)),
            other => anyhow::bail!("Unknown quantization level: '{}'. Use q4_0 or q8_0.", other),
        },
    }
}

/// High-level GLM-OCR engine.
///
/// Usage:
/// ```no_run
/// let ocr = GlmOcr::new(None, None)?;
/// let text = ocr.recognize(&image, "Text Recognition:")?;
/// ```
pub struct GlmOcr {
    model: GlmOcrModel,
    tokenizer: GlmOcrTokenizer,
    #[allow(dead_code)]
    config: GlmOcrConfig,
}

impl GlmOcr {
    /// Create a new GLM-OCR engine on CPU.
    ///
    /// Downloads the model from HuggingFace on first use (~2.65GB).
    /// Set `model_id` to use a custom model (default: "unsloth/GLM-OCR").
    /// Set `quantize` to a quantization level string ("q8_0", "q4_0") or None.
    pub fn new(model_id: Option<&str>, quantize: Option<&str>) -> Result<Self> {
        Self::new_with_device(model_id, quantize, Device::Cpu)
    }

    /// Create a new GLM-OCR engine on a specific device (CPU or CUDA GPU).
    ///
    /// For GPU: use `Device::new_cuda(0)?` (requires the `cuda` cargo feature).
    /// On GPU, F16 is used for faster matmul; on CPU, F32 is used.
    /// Set `quantize` to "q8_0" or "q4_0" for quantized inference.
    pub fn new_with_device(model_id: Option<&str>, quantize: Option<&str>, device: Device) -> Result<Self> {
        let qdtype = parse_quantization(quantize)?;

        // Use F16 on GPU for faster inference, F32 on CPU (candle CPU lacks BF16/F16 matmul)
        // Skip quantization on GPU — QMatMul requires F32 so every layer would need
        // F16→F32→F32→F16 casts, which is slower than native F16 matmul.
        let (dtype, qdtype) = if device.is_cuda() {
            if qdtype.is_some() {
                tracing::info!("GPU detected — skipping quantization (native F16 is faster)");
            }
            (DType::F16, None)
        } else {
            (DType::F32, qdtype)
        };

        let loader = ModelLoader::new(model_id);

        tracing::info!("Loading config...");
        let config = loader.load_config()?;

        tracing::info!("Loading tokenizer...");
        let tokenizer_path = loader.tokenizer_path()?;
        let tokenizer = GlmOcrTokenizer::from_file(&tokenizer_path, &config)?;

        tracing::info!("Loading model weights (~2.65GB)...");
        let vb = loader.load_weights(dtype, &device)?;

        match qdtype {
            Some(GgmlDType::Q4_0) => tracing::info!("Building model with Q4_0 quantization..."),
            Some(GgmlDType::Q8_0) => tracing::info!("Building model with Q8_0 quantization..."),
            Some(q) => tracing::info!("Building model with {:?} quantization...", q),
            None => tracing::info!("Building model..."),
        }
        let model = GlmOcrModel::new(&config, vb, &device, dtype, qdtype)?;

        tracing::info!("Model ready on {:?}.", device);
        Ok(Self {
            model,
            tokenizer,
            config,
        })
    }

    /// Recognize text from an image.
    ///
    /// For large images, automatically splits into tiles and processes each piece.
    pub fn recognize(&self, image: &DynamicImage, prompt: &str) -> Result<String> {
        self.recognize_with_max_tokens(image, prompt, 8192)
    }

    /// Recognize text with a custom max token limit.
    ///
    /// Large images are automatically split into horizontal strips to limit memory.
    pub fn recognize_with_max_tokens(
        &self,
        image: &DynamicImage,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let (w, h) = (image.width(), image.height());
        let unit = 28u32; // patch_size(14) * merge_size(2)

        // Compute actual merged patch count to decide if stripping is needed
        let w_merged = ((w + unit / 2) / unit).max(1);
        let h_merged = ((h + unit / 2) / unit).max(1);
        let total_merged = w_merged * h_merged;

        if total_merged > MAX_MERGED_PATCHES {
            self.recognize_stripped(image, prompt, max_tokens)
        } else {
            generation::generate(&self.model, &self.tokenizer, image, prompt, max_tokens)
        }
    }

    /// Process a large image by splitting into horizontal strips.
    ///
    /// Each strip spans the full image width and has limited height computed
    /// from the patch budget. No width scaling — preserves text readability.
    fn recognize_stripped(
        &self,
        image: &DynamicImage,
        prompt: &str,
        max_tokens_per_strip: usize,
    ) -> Result<String> {
        let (img_w, img_h) = (image.width(), image.height());
        let unit = 28u32; // patch_size * merge_size

        // Width in merged patches (after rounding to unit)
        let w_patches = ((img_w + unit / 2) / unit).max(1);
        let merged_w = w_patches / 2; // merge_size = 2

        // Max strip height in merged patches that fits within budget
        let max_h_merged = MAX_MERGED_PATCHES / merged_w;
        let max_strip_h = max_h_merged * 2 * 14; // merged * merge_size * patch_size

        let stride = max_strip_h.saturating_sub(STRIP_OVERLAP);
        let num_strips = if img_h <= max_strip_h {
            1
        } else {
            ((img_h - STRIP_OVERLAP) + stride - 1) / stride
        };

        tracing::info!(
            "Stripping {}x{} image into {} horizontal strips (strip_h={}px, ~{} merged patches/strip)",
            img_w, img_h, num_strips, max_strip_h, merged_w * max_h_merged
        );

        let mut results = Vec::new();

        for i in 0..num_strips {
            let y = (i * stride).min(img_h.saturating_sub(max_strip_h));
            let h = max_strip_h.min(img_h - y);

            tracing::info!(
                "Processing strip {}/{} at y={} h={}",
                i + 1, num_strips, y, h
            );

            let strip = image.crop_imm(0, y, img_w, h);

            match generation::generate(
                &self.model,
                &self.tokenizer,
                &strip,
                prompt,
                max_tokens_per_strip,
            ) {
                Ok(text) => {
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        results.push(text);
                    }
                }
                Err(e) => {
                    tracing::warn!("Strip {}/{} failed: {}", i + 1, num_strips, e);
                }
            }
        }

        Ok(results.join("\n"))
    }

    /// Recognize text using layout detection to intelligently segment the document.
    ///
    /// Pipeline:
    /// 1. Run PP-DocLayout-M layout detection on the full page
    /// 2. Sort detections by reading order (top-to-bottom, left-to-right)
    /// 3. Crop each detected region
    /// 4. Run GLM-OCR on each region with appropriate prompt
    /// 5. Assemble results into markdown with region-type-aware formatting
    pub fn recognize_with_layout(
        &self,
        image: &DynamicImage,
        layout: &mut LayoutDetector,
        max_tokens_per_region: usize,
    ) -> Result<String> {
        let doc = self.recognize_layout_structured(image, layout, max_tokens_per_region)?;
        Ok(doc.to_markdown())
    }

    /// Recognize text using layout detection, returning structured data.
    ///
    /// Returns a [`DocumentLayout`] with typed sections preserving the layout
    /// label, bounding box, raw text, extracted key-value pairs, and parsed
    /// table data. Use [`DocumentLayout::to_markdown()`] or
    /// [`DocumentLayout::to_json()`] to render the output.
    pub fn recognize_layout_structured(
        &self,
        image: &DynamicImage,
        layout: &mut LayoutDetector,
        max_tokens_per_region: usize,
    ) -> Result<DocumentLayout> {
        let rgb = image.to_rgb8();
        let (img_w, img_h) = (rgb.width(), rgb.height());
        let detections = layout.detect(&rgb)?;

        if detections.is_empty() {
            tracing::warn!("No layout regions detected, falling back to strip-based processing");
            let text = self.recognize_with_max_tokens(image, "Text Recognition:", max_tokens_per_region)?;
            return Ok(DocumentLayout {
                width: img_w,
                height: img_h,
                sections: vec![DocumentSection {
                    label: "text".to_string(),
                    bbox: [0.0, 0.0, img_w as f32, img_h as f32],
                    text: text.clone(),
                    key_values: extract_key_values(&text),
                    table: None,
                }],
            });
        }

        tracing::info!("Detected {} layout regions", detections.len());
        for det in &detections {
            tracing::info!(
                "  {} (score={:.2}) at [{:.0}, {:.0}, {:.0}, {:.0}]",
                det.label, det.score, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
            );
        }

        // Merge adjacent same-label text blocks to reduce VLM calls
        let mut merged = merge_adjacent_blocks(&detections);
        tracing::info!(
            "Merged {} regions into {} groups",
            detections.len(),
            merged.len()
        );

        // Fill vertical gaps not covered by any detected region.
        let gap_regions = find_vertical_gaps(&merged, img_w as f32, img_h as f32);
        if !gap_regions.is_empty() {
            tracing::info!("Adding {} gap-fill regions", gap_regions.len());
            for gap in &gap_regions {
                tracing::info!(
                    "  gap text at [{:.0}, {:.0}, {:.0}, {:.0}]",
                    gap.bbox[0], gap.bbox[1], gap.bbox[2], gap.bbox[3]
                );
            }
            let mut all_regions = Vec::new();
            let mut gap_iter = gap_regions.into_iter().peekable();
            for region in merged {
                while let Some(gap) = gap_iter.peek() {
                    if gap.bbox[1] < region.bbox[1] {
                        all_regions.push(gap_iter.next().unwrap());
                    } else {
                        break;
                    }
                }
                all_regions.push(region);
            }
            all_regions.extend(gap_iter);
            merged = all_regions;
        }

        let mut sections: Vec<DocumentSection> = Vec::new();
        let pad = 4; // padding pixels around crop

        for (i, region) in merged.iter().enumerate() {
            if IMAGE_LABELS.contains(&region.label) {
                tracing::info!(
                    "Skipping region {}/{}: {} (image label)",
                    i + 1,
                    merged.len(),
                    region.label,
                );
                continue;
            }

            let x1 = (region.bbox[0] as u32).saturating_sub(pad);
            let y1 = (region.bbox[1] as u32).saturating_sub(pad);
            let x2 = ((region.bbox[2] as u32) + pad).min(img_w);
            let y2 = ((region.bbox[3] as u32) + pad).min(img_h);
            let w = x2 - x1;
            let h = y2 - y1;

            if w < 10 || h < 10 {
                continue;
            }

            let crop = image.crop_imm(x1, y1, w, h);
            let crop = scale_to_patch_budget(&crop, MAX_MERGED_PATCHES);

            let prompt = prompt_for_label(region.label);

            tracing::info!(
                "Processing region {}/{}: {} ({}x{}) prompt=\"{}\"",
                i + 1,
                merged.len(),
                region.label,
                crop.width(),
                crop.height(),
                prompt,
            );

            match self.recognize_with_max_tokens(&crop, prompt, max_tokens_per_region) {
                Ok(text) => {
                    let text = truncate_repetitive_content(text.trim());
                    let text = strip_empty_code_blocks(&text);
                    if !text.is_empty() {
                        // Parse table data for table regions
                        let table = if region.label == "table" {
                            let (td, _) = parse_table_region(&text);
                            td
                        } else {
                            None
                        };

                        // Extract key-value pairs from text-like regions
                        let key_values = if region.label != "table" {
                            extract_key_values(&text)
                        } else {
                            Vec::new()
                        };

                        sections.push(DocumentSection {
                            label: region.label.to_string(),
                            bbox: region.bbox,
                            text,
                            key_values,
                            table,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Region {}/{} ({}) failed: {}",
                        i + 1,
                        merged.len(),
                        region.label,
                        e
                    );
                }
            }
        }

        Ok(DocumentLayout {
            width: img_w,
            height: img_h,
            sections,
        })
    }
}

/// Labels that should skip VLM OCR entirely (images, seals, decorative elements).
const IMAGE_LABELS: &[&str] = &["image", "header_image", "footer_image", "seal"];

/// Labels that should NOT be merged with adjacent blocks.
const NON_MERGE_LABELS: &[&str] = &[
    "image", "header_image", "footer_image", "seal", "table", "chart", "formula",
];

/// A merged group of layout regions to process as one VLM call.
struct MergedRegion {
    /// Combined bounding box encompassing all blocks in the group.
    bbox: [f32; 4],
    /// Label of the group (from the first block).
    label: &'static str,
    /// Score of the highest-confidence block in the group.
    score: f32,
}

/// Merge adjacent same-label text blocks into groups.
///
/// Blocks that are vertically adjacent (horizontal projection overlap > 0),
/// same label, and close together (gap < 50% of taller block height) get merged.
/// Non-merge labels (tables, images, formulas, charts) are kept separate.
fn merge_adjacent_blocks(detections: &[layout::LayoutDetection]) -> Vec<MergedRegion> {
    let mut result: Vec<MergedRegion> = Vec::new();

    for det in detections {
        if NON_MERGE_LABELS.contains(&det.label) || IMAGE_LABELS.contains(&det.label) {
            // Non-mergeable: keep as its own region
            result.push(MergedRegion {
                bbox: det.bbox,
                label: det.label,
                score: det.score,
            });
            continue;
        }

        // Try to merge with the last region if compatible
        let should_merge = if let Some(last) = result.last() {
            last.label == det.label
                && !NON_MERGE_LABELS.contains(&last.label)
                && blocks_are_adjacent(&last.bbox, &det.bbox)
        } else {
            false
        };

        if should_merge {
            let last = result.last_mut().unwrap();
            // Expand bbox to encompass both
            last.bbox[0] = last.bbox[0].min(det.bbox[0]);
            last.bbox[1] = last.bbox[1].min(det.bbox[1]);
            last.bbox[2] = last.bbox[2].max(det.bbox[2]);
            last.bbox[3] = last.bbox[3].max(det.bbox[3]);
            last.score = last.score.max(det.score);
        } else {
            result.push(MergedRegion {
                bbox: det.bbox,
                label: det.label,
                score: det.score,
            });
        }
    }

    result
}

/// Find vertical gaps in detected regions and create synthetic "text" regions
/// to fill them. This catches content that the layout detector missed (headers,
/// inter-region text, etc.).
///
/// Only creates gap regions that are at least `MIN_GAP_HEIGHT` pixels tall and
/// span at least half the page width.
fn find_vertical_gaps(
    regions: &[MergedRegion],
    page_w: f32,
    page_h: f32,
) -> Vec<MergedRegion> {
    const MIN_GAP_HEIGHT: f32 = 20.0;

    if regions.is_empty() {
        return Vec::new();
    }

    // Collect all y-intervals from regions (including image labels, since they cover area)
    let mut intervals: Vec<(f32, f32)> = regions
        .iter()
        .map(|r| (r.bbox[1], r.bbox[3]))
        .collect();
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Merge overlapping intervals
    let mut merged_intervals: Vec<(f32, f32)> = Vec::new();
    for (y1, y2) in intervals {
        if let Some(last) = merged_intervals.last_mut() {
            if y1 <= last.1 {
                last.1 = last.1.max(y2);
                continue;
            }
        }
        merged_intervals.push((y1, y2));
    }

    // Find gaps
    let mut gaps = Vec::new();

    // Gap at the top of the page
    if let Some(&(first_y1, _)) = merged_intervals.first() {
        if first_y1 > MIN_GAP_HEIGHT {
            gaps.push(MergedRegion {
                bbox: [0.0, 0.0, page_w, first_y1],
                label: "text",
                score: 0.0,
            });
        }
    }

    // Gaps between regions
    for pair in merged_intervals.windows(2) {
        let gap_y1 = pair[0].1;
        let gap_y2 = pair[1].0;
        let gap_h = gap_y2 - gap_y1;
        if gap_h >= MIN_GAP_HEIGHT {
            gaps.push(MergedRegion {
                bbox: [0.0, gap_y1, page_w, gap_y2],
                label: "text",
                score: 0.0,
            });
        }
    }

    // Gap at the bottom (skip — usually blank margin)

    gaps
}

/// Check if two blocks are vertically adjacent and horizontally aligned.
fn blocks_are_adjacent(a: &[f32; 4], b: &[f32; 4]) -> bool {
    // Horizontal projection overlap: do they share x-range?
    let h_overlap_start = a[0].max(b[0]);
    let h_overlap_end = a[2].min(b[2]);
    let h_overlap = h_overlap_end - h_overlap_start;
    if h_overlap <= 0.0 {
        return false;
    }

    // Vertical gap: b should be below a (b.y1 >= a.y1 since sorted by reading order)
    let gap = (b[1] - a[3]).max(0.0);
    let max_height = (a[3] - a[1]).max(b[3] - b[1]);

    // Gap must be less than 50% of the taller block's height
    gap < max_height * 0.5
}

/// Scale an image down so its merged patch count fits within the budget.
/// Preserves aspect ratio. Returns the original image if it already fits.
fn scale_to_patch_budget(image: &DynamicImage, max_patches: u32) -> DynamicImage {
    let unit = 28u32; // patch_size(14) * merge_size(2)
    let w = image.width();
    let h = image.height();

    let w_merged = ((w + unit / 2) / unit).max(1);
    let h_merged = ((h + unit / 2) / unit).max(1);
    let total = w_merged * h_merged;

    if total <= max_patches {
        return image.clone();
    }

    // Scale factor to fit within budget
    let scale = (max_patches as f64 / total as f64).sqrt();
    let new_w = ((w as f64 * scale) as u32).max(unit);
    let new_h = ((h as f64 * scale) as u32).max(unit);

    tracing::info!(
        "Scaling {}x{} ({} patches) → {}x{} to fit {} patch budget",
        w, h, total, new_w, new_h, max_patches,
    );

    image.resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
}

/// Choose an appropriate OCR prompt based on the layout region type.
/// GLM-OCR supports "Text Recognition:", "Formula Recognition:", "Table Recognition:"
/// per https://huggingface.co/zai-org/GLM-OCR docs.
/// "Table Recognition:" outputs HTML tables but is much slower on CPU due to verbose
/// output. Use "Text Recognition:" for layout mode; users can pass "Table Recognition:"
/// directly via --prompt for single-image mode.
fn prompt_for_label(label: &str) -> &str {
    match label {
        "formula" => "Formula Recognition:",
        // Table regions often include surrounding text (headers, totals) in layout
        // detection. "Text Recognition:" captures everything; we format the tabular
        // portion as markdown in post-processing.
        _ => "Text Recognition:",
    }
}

/// Detect and truncate repetitive content from VLM output.
///
/// Handles three types of hallucination:
/// 1. Phrase-level suffix repetition in long single lines
/// 2. Full-string character-level repetition (e.g., "abcabc")
/// 3. Line-level repetition (same line repeated many times)
fn truncate_repetitive_content(content: &str) -> String {
    let stripped = content.trim();
    if stripped.is_empty() {
        return content.to_string();
    }

    // Priority 1: Phrase-level suffix repetition in long single lines
    if !stripped.contains('\n') && stripped.len() > 100 {
        if let Some((prefix, _, count)) = find_repeating_suffix(stripped, 8, 5) {
            let unit_len = stripped.len() - prefix.len();
            if unit_len > 0 {
                let repeat_portion = (unit_len / count) * count;
                if repeat_portion as f64 > stripped.len() as f64 * 0.5 {
                    return prefix.to_string();
                }
            }
        }
    }

    // Priority 2: Full-string character-level repetition
    if !stripped.contains('\n') && stripped.len() > 10 {
        if let Some(unit) = find_shortest_repeating_substring(stripped) {
            let count = stripped.len() / unit.len();
            if count >= 10 {
                return unit.to_string();
            }
        }
    }

    // Priority 3: Line-level repetition
    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() >= 10 {
        let mut counts = std::collections::HashMap::new();
        for line in &lines {
            *counts.entry(*line).or_insert(0u32) += 1;
        }
        if let Some((&most_common, &count)) = counts.iter().max_by_key(|(_, v)| **v) {
            if count >= 10 && (count as f64 / lines.len() as f64) >= 0.8 {
                return most_common.to_string();
            }
        }
    }

    content.to_string()
}

/// Find the shortest substring that repeats to form the entire string.
fn find_shortest_repeating_substring(s: &str) -> Option<String> {
    let n = s.len();
    for i in 1..=n / 2 {
        if n % i == 0 {
            let unit = &s[..i];
            let count = n / i;
            if unit.repeat(count) == s {
                return Some(unit.to_string());
            }
        }
    }
    None
}

/// Detect if a string ends with a repeating phrase.
/// Returns (prefix, unit, count) if found.
fn find_repeating_suffix(s: &str, min_len: usize, min_repeats: usize) -> Option<(&str, &str, usize)> {
    let len = s.len();
    let max_unit = len / min_repeats;
    for i in (min_len..=max_unit).rev() {
        if i > len {
            continue;
        }
        let unit = &s[len - i..];
        let repeated = unit.repeat(min_repeats);
        if s.ends_with(&repeated) {
            // Count total repeats from the end
            let mut count = 0;
            let mut pos = len;
            while pos >= i && &s[pos - i..pos] == unit {
                pos -= i;
                count += 1;
            }
            let prefix_end = len - (count * i);
            return Some((&s[..prefix_end], unit, count));
        }
    }
    None
}

/// Format OCR text based on the layout region type.
fn format_region_by_label(text: &str, label: &str) -> String {
    match label {
        "table" => format_table_region(text),
        "doc_title" | "paragraph_title" => format!("## {text}"),
        "table_title" | "chart_title" | "figure_title" => format!("### {text}"),
        "header" | "header_image" => format!("*{text}*"),
        "footer" | "footer_image" | "footnote" => format!("_{text}_"),
        _ => text.to_string(),
    }
}

/// Known column headers that appear in purchase order / invoice tables.
/// Used to detect where the tabular data starts in "Text Recognition:" output.
const TABLE_HEADER_KEYWORDS: &[&str] = &[
    "Item", "Article", "Barcode", "Quantity", "UOM", "Gross", "Excise",
    "Tax", "Total", "Net", "Brand", "Description", "Reference", "Price",
    "Amount", "Qty", "Unit", "Rate", "Discount", "Value", "S.No", "Sr.",
    "Sl.", "No.", "HSN", "SAC", "CGST", "SGST", "IGST",
];

/// Format a table region's "Text Recognition:" output as markdown.
fn format_table_region(text: &str) -> String {
    let (_, markdown) = parse_table_region(text);
    markdown
}

/// Parse a table region's "Text Recognition:" output.
///
/// Returns `(Option<TableData>, markdown_string)`. The TableData is populated
/// when a tabular header with 4+ known keywords is detected. The markdown
/// string is always the full formatted output (prefix + table + suffix).
fn parse_table_region(text: &str) -> (Option<TableData>, String) {
    let lines: Vec<&str> = text.lines().collect();

    // Find the table header line: a line containing 4+ known column keywords
    let header_idx = lines.iter().position(|line| {
        let keyword_count = TABLE_HEADER_KEYWORDS
            .iter()
            .filter(|kw| line.contains(**kw))
            .count();
        keyword_count >= 4
    });

    let header_idx = match header_idx {
        Some(idx) => idx,
        None => return (None, text.to_string()),
    };

    // Find data rows: lines after header that start with a number or item code,
    // and stop at blank lines or "Total" summary lines.
    let mut data_end = header_idx + 1;
    for i in (header_idx + 1)..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || trimmed.starts_with("Total") {
            break;
        }
        data_end = i + 1;
    }

    // Parse header and data into structured form
    let header_line = lines[header_idx];
    let data_lines: Vec<&str> = lines[header_idx + 1..data_end].to_vec();

    let (table_data, md_table) = parse_and_format_table(header_line, &data_lines);

    // Build markdown output: prefix text + markdown table + suffix text
    let mut result = Vec::new();

    let prefix: String = lines[..header_idx]
        .iter()
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    if !prefix.trim().is_empty() {
        result.push(prefix);
    }

    result.push(md_table);

    let suffix: String = lines[data_end..]
        .iter()
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    if !suffix.trim().is_empty() {
        result.push(suffix);
    }

    (table_data, result.join("\n\n"))
}

/// Known column name patterns for splitting headers.
/// Order matters: longer/more-specific patterns first.
const KNOWN_COLUMNS: &[&str] = &[
    "Article Description + Add.Info",
    "Article Description + Add Info",
    "Article Description",
    "Total Value (WT)",
    "Total Value (Wt)",
    "Supplier Reference",
    "Total Value",
    "Total Amount",
    "Excise Value",
    "Excise /PU",
    "Gross Amount",
    "Net W/Ex.T",
    "Net WIEx.T",
    "Net Wt.E.T",
    "Net W/E.T",
    "Net Amount",
    "Excise PU",
    "Tax Value",
    "Tax Rate",
    "Sr. No",
    "Sl. No",
    "HSN/SAC",
    "Gross/PU",
    "Tax Val",
    "Tax %",
    "Description",
    "Reference",
    "Quantity",
    "Barcode",
    "Discount",
    "Article",
    "Brand",
    "Price",
    "Amount",
    "Item",
    "S.No",
    "Rate",
    "CGST",
    "SGST",
    "IGST",
    "UOM",
    "Qty",
    "HSN",
];

/// Parse a plain-text header + data rows into structured TableData and markdown.
///
/// Returns `(Option<TableData>, markdown_string)`. TableData is None when
/// there are no data rows or the header has fewer than 3 columns.
fn parse_and_format_table(header: &str, data_rows: &[&str]) -> (Option<TableData>, String) {
    if data_rows.is_empty() {
        return (None, header.to_string());
    }

    let header_cols = split_header_left_to_right(header);

    if header_cols.len() < 3 {
        return (None, format!("{}\n{}", header, data_rows.join("\n")));
    }

    let n_cols = header_cols.len();

    let is_text_col: Vec<bool> = header_cols
        .iter()
        .map(|col| is_description_column(col))
        .collect();

    // Build structured rows and markdown simultaneously
    let mut md_lines = Vec::new();
    md_lines.push(format!("| {} |", header_cols.join(" | ")));
    md_lines.push(format!(
        "| {} |",
        header_cols.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")
    ));

    let mut structured_rows = Vec::new();

    for row in data_rows {
        let words: Vec<&str> = row.split_whitespace().collect();
        let cells = align_data_to_columns(&words, &is_text_col, n_cols);
        md_lines.push(format!("| {} |", cells.join(" | ")));
        structured_rows.push(cells);
    }

    let table_data = TableData {
        headers: header_cols,
        rows: structured_rows,
    };

    (Some(table_data), md_lines.join("\n"))
}

/// Parse header text into column names by greedily matching known patterns
/// from left to right.
fn split_header_left_to_right(header: &str) -> Vec<String> {
    let mut cols: Vec<String> = Vec::new();
    let mut remaining = header.trim();

    while !remaining.is_empty() {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }

        let mut matched = false;
        for pattern in KNOWN_COLUMNS {
            if remaining.starts_with(pattern) {
                let after = &remaining[pattern.len()..];
                // Must be followed by space, end of string, or another column
                if after.is_empty() || after.starts_with(' ') {
                    cols.push(pattern.to_string());
                    remaining = after;
                    matched = true;
                    break;
                }
            }
        }

        if !matched {
            // Take the next word as an unknown column name
            let end = remaining.find(' ').unwrap_or(remaining.len());
            cols.push(remaining[..end].to_string());
            remaining = &remaining[end..];
        }
    }

    cols
}

/// Align data row words to header columns.
///
/// Assigns words from both ends inward: single-word columns at the start get
/// one word each (left-to-right), single-word columns from the end also get
/// one word each (right-to-left). All remaining middle words go to the largest
/// multi-word "description" column. Other multi-word columns get empty strings.
fn align_data_to_columns(words: &[&str], is_text_col: &[bool], n_cols: usize) -> Vec<String> {
    let n_words = words.len();

    if n_words == 0 || n_cols == 0 {
        return vec![String::new(); n_cols];
    }

    let mut cells = vec![String::new(); n_cols];

    // Find the longest multi-word header column — that's the description column
    // which should absorb all remaining words.
    let desc_col_idx = is_text_col
        .iter()
        .enumerate()
        .filter(|(_, &t)| t)
        .max_by_key(|(i, _)| {
            // Prefer the column with the longest header name
            // (Description columns tend to have long names)
            *i // tie-break: pick later column
        })
        .map(|(i, _)| i);

    // Assign from left: single-word columns get one word each
    let mut left_wi = 0;
    let mut left_assigned = vec![false; n_cols];
    for ci in 0..n_cols {
        if ci == desc_col_idx.unwrap_or(usize::MAX) {
            break; // stop at description column
        }
        if !is_text_col[ci] {
            if left_wi < n_words {
                cells[ci] = words[left_wi].to_string();
                left_wi += 1;
                left_assigned[ci] = true;
            }
        }
        // Multi-word columns before description get empty (e.g., Supplier Reference)
    }

    // Assign from right: single-word columns get one word each
    let mut right_wi = n_words;
    let mut right_assigned = vec![false; n_cols];
    for ci in (0..n_cols).rev() {
        if ci == desc_col_idx.unwrap_or(usize::MAX) {
            break;
        }
        if left_assigned[ci] {
            continue; // already filled from the left
        }
        if !is_text_col[ci] {
            if right_wi > left_wi {
                right_wi -= 1;
                cells[ci] = words[right_wi].to_string();
                right_assigned[ci] = true;
            }
        }
    }

    // Remaining middle words go to the description column
    if let Some(di) = desc_col_idx {
        if left_wi < right_wi {
            cells[di] = words[left_wi..right_wi].join(" ");
        }
    }

    cells
}

/// Check if a column header name represents a free-text description field
/// (which may consume multiple data words). Most columns are "value" columns
/// that take exactly one data word each.
fn is_description_column(name: &str) -> bool {
    // Known description-like column patterns
    let lower = name.to_lowercase();
    lower.contains("description")
        || lower.contains("particulars")
        || lower.contains("narration")
        || lower == "supplier reference"
        || lower == "reference"
        || lower == "remarks"
        || lower == "address"
}

/// Check if a token is a "simple" single-word value (numeric, UOM, or percentage).
fn is_simple_value(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    // UOM codes
    if is_uom_token(s) {
        return true;
    }
    // Percentage: "5.00%"
    if s.ends_with('%') {
        return s[..s.len() - 1].replace(',', "").parse::<f64>().is_ok();
    }
    // Number with optional commas: "4,041.00", "0.00", "1", "4242003788899"
    let cleaned: String = s.chars().filter(|c| *c != ',').collect();
    cleaned.parse::<f64>().is_ok()
}

/// Check if a short token is a unit of measure.
fn is_uom_token(s: &str) -> bool {
    matches!(
        s,
        "EA" | "PC" | "PCS" | "KG" | "LTR" | "MTR" | "BOX" | "CTN" | "DZ" | "SET" | "PKT"
            | "NOS" | "PAR" | "PRS" | "LT" | "ML" | "GM" | "EACH"
    )
}

/// Extract key-value pairs from text lines matching "Key: Value" patterns.
///
/// Heuristics: the key must be at most 40 chars and 5 words, and the value
/// must be non-empty. This catches common document fields like
/// "Order Date: 16/03/2020" without false-positiving on prose sentences.
fn extract_key_values(text: &str) -> Vec<KeyValue> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let colon_pos = line.find(':')?;
            if colon_pos == 0 || colon_pos > 40 {
                return None;
            }
            let key = line[..colon_pos].trim();
            let value = line[colon_pos + 1..].trim();
            if key.split_whitespace().count() > 5 {
                return None;
            }
            if value.is_empty() {
                return None;
            }
            Some(KeyValue {
                key: key.to_string(),
                value: value.to_string(),
            })
        })
        .collect()
}

/// Strip empty markdown code blocks from OCR output.
///
/// The VLM sometimes outputs ` ```markdown\n\n``` ` for image-only regions
/// (logos, decorative elements). After stripping, the result is empty and
/// the section gets filtered out.
fn strip_empty_code_blocks(text: &str) -> String {
    // Remove empty fenced code blocks: ```<optional-lang>\n<whitespace>\n```
    let mut result = text.to_string();
    loop {
        // Match ```<word>\n<whitespace>\n```  or  ```\n<whitespace>\n```
        if let Some(start) = result.find("```") {
            let after_backticks = &result[start + 3..];
            // Skip optional language tag
            let rest = if let Some(nl) = after_backticks.find('\n') {
                let lang_part = &after_backticks[..nl];
                if lang_part.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
                    &after_backticks[nl + 1..]
                } else {
                    break;
                }
            } else {
                break;
            };
            // Check if remaining is just whitespace + closing ```
            if let Some(close_pos) = rest.find("```") {
                let between = &rest[..close_pos];
                if between.trim().is_empty() {
                    let end = start + 3 + (rest.as_ptr() as usize - after_backticks.as_ptr() as usize) + close_pos + 3;
                    result = format!("{}{}", &result[..start], result[end..].trim_start());
                    continue;
                }
            }
        }
        break;
    }
    result.trim().to_string()
}

/// Deduplicate lines that appear at the tail of one section and the head
/// of the next. This handles overlap from gap-fill regions.
fn deduplicate_adjacent_sections(sections: &[String]) -> String {
    if sections.is_empty() {
        return String::new();
    }
    if sections.len() == 1 {
        return sections[0].clone();
    }

    let mut result = sections[0].clone();

    for section in &sections[1..] {
        let prev_lines: Vec<&str> = result.lines().collect();
        let next_lines: Vec<&str> = section.lines().collect();

        // Find the longest suffix of prev that matches a prefix of next
        let max_overlap = prev_lines.len().min(next_lines.len()).min(5);
        let mut overlap = 0;

        for n in (1..=max_overlap).rev() {
            let prev_tail = &prev_lines[prev_lines.len() - n..];
            let next_head = &next_lines[..n];
            if prev_tail.iter().zip(next_head.iter()).all(|(a, b)| {
                a.trim() == b.trim() && !a.trim().is_empty()
            }) {
                overlap = n;
                break;
            }
        }

        if overlap > 0 {
            // Skip the overlapping lines from the next section
            let deduped: String = next_lines[overlap..].join("\n");
            result.push_str("\n\n");
            result.push_str(deduped.trim_start());
        } else {
            result.push_str("\n\n");
            result.push_str(section);
        }
    }

    result
}
