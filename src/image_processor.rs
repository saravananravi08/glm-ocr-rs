use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;

use crate::config::{VisionConfig, IMAGE_MEAN, IMAGE_STD};

/// Preprocess an image for GLM-OCR's vision encoder.
///
/// Returns:
/// - `pixel_values`: `[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]`
/// - `image_grid_thw`: `[1, 3]` tensor `[temporal, grid_h, grid_w]`
pub fn preprocess_image(
    img: &DynamicImage,
    config: &VisionConfig,
    device: &Device,
) -> Result<(Tensor, [u32; 3])> {
    let patch_size = config.patch_size as u32;
    let merge_size = config.spatial_merge_size as u32;
    let unit = patch_size * merge_size; // 28

    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    // Compute target size: ensure dimensions are multiples of unit (28),
    // and total patch count stays within limits.
    let (target_h, target_w) = compute_target_size(orig_h, orig_w, unit);

    // Resize
    let resized = image::imageops::resize(
        &rgb,
        target_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    );

    let grid_h = target_h / patch_size;
    let grid_w = target_w / patch_size;
    let temporal = 1u32;

    // Convert to float tensor [C, H, W], normalize
    let mut pixel_data =
        vec![0f32; (config.in_channels * target_h as usize * target_w as usize) as usize];

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x, y);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - IMAGE_MEAN[c]) / IMAGE_STD[c];
                let idx = c as usize * (target_h as usize * target_w as usize)
                    + y as usize * target_w as usize
                    + x as usize;
                pixel_data[idx] = normalized;
            }
        }
    }

    // Reshape into patches for Conv3d: each patch is [C, temporal_patch_size, patch_size, patch_size]
    // Since we have a single image (T=1) and temporal_patch_size=2, we pad the temporal dim with zeros.
    // The Conv3d expects input grouped into chunks of [C, T, pH, pW].
    //
    // For a single image: we treat it as T=temporal_patch_size frames (duplicate or zero-pad).
    // Actually, the HF processor for single images sets temporal=1 in grid_thw and the
    // patch_embed groups temporal_patch_size frames together. For T=1 with temporal_patch_size=2,
    // we need to check: num_temporal_patches = ceil(T / temporal_patch_size).
    // With T=1 and temporal_patch_size=2, num_temporal_patches = 1, and the patch is zero-padded.
    //
    // Total patches = num_temporal_patches * grid_h * grid_w = 1 * grid_h * grid_w
    let num_patches = (grid_h * grid_w) as usize;
    let patch_volume =
        config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;

    let mut patches = vec![0f32; num_patches * patch_volume];

    // Fill patches in MERGE-GROUPED order matching HuggingFace's 9D reshape+transpose:
    //   reshape(grid_t, T_ps, C, grid_h//ms, ms, pH, grid_w//ms, ms, pW)
    //   transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    // Result ordering: for each merge-block (by, bx), for each sub-position (my, mx),
    // the patch at grid position (by*merge+my, bx*merge+mx).
    //
    // Within each patch, data layout is [C, T, pH, pW] (C-contiguous).
    // Temporal padding: duplicate the single frame for both t=0 and t=1.
    let merge = merge_size as usize;
    let bh = grid_h as usize / merge;
    let bw = grid_w as usize / merge;
    let mut patch_idx = 0usize;

    for by in 0..bh {
        for bx in 0..bw {
            for my in 0..merge {
                for mx in 0..merge {
                    let ph = by * merge + my;
                    let pw = bx * merge + mx;
                    let base = patch_idx * patch_volume;

                    for c in 0..config.in_channels {
                        for py in 0..config.patch_size {
                            for px in 0..config.patch_size {
                                let img_y = ph * config.patch_size + py;
                                let img_x = pw * config.patch_size + px;
                                let src_idx =
                                    c * (target_h as usize * target_w as usize)
                                        + img_y * target_w as usize
                                        + img_x;
                                let pixel_val = pixel_data[src_idx];

                                // Index into patch: [C, T, pH, pW]
                                // Duplicate frame for both temporal slots
                                for t in 0..config.temporal_patch_size {
                                    let dst_idx = c
                                        * (config.temporal_patch_size * config.patch_size * config.patch_size)
                                        + t * (config.patch_size * config.patch_size)
                                        + py * config.patch_size
                                        + px;
                                    patches[base + dst_idx] = pixel_val;
                                }
                            }
                        }
                    }

                    patch_idx += 1;
                }
            }
        }
    }

    let pixel_values =
        Tensor::from_vec(patches, (num_patches, patch_volume), device)?.to_dtype(DType::F32)?;

    let grid_thw = [temporal, grid_h, grid_w];

    Ok((pixel_values, grid_thw))
}

/// Compute target image dimensions that are multiples of `unit` (patch_size * merge_size = 28).
/// Maintains aspect ratio, ensures total patch count is reasonable.
fn compute_target_size(orig_h: u32, orig_w: u32, unit: u32) -> (u32, u32) {
    // Max total patches: 12544 (from preprocessor_config shortest_edge)
    // That's sqrt(12544) ≈ 112 patches per side at patch_size=14 → 1568 pixels
    let max_patches: u32 = 12544;
    let patch_size = unit / 2; // 14 (unit = patch_size * merge_size)

    // Round to nearest unit
    let mut target_h = ((orig_h + unit / 2) / unit).max(1) * unit;
    let mut target_w = ((orig_w + unit / 2) / unit).max(1) * unit;

    // Check total patches doesn't exceed limit
    let total_patches = (target_h / patch_size) * (target_w / patch_size);
    if total_patches > max_patches {
        let scale = (max_patches as f64 / total_patches as f64).sqrt();
        target_h = ((target_h as f64 * scale) as u32 / unit).max(1) * unit;
        target_w = ((target_w as f64 * scale) as u32 / unit).max(1) * unit;
    }

    (target_h, target_w)
}
