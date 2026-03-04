use candle_core::{Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, RmsNorm, VarBuilder};

use crate::config::VisionConfig;
use crate::nn_utils::rms_norm;
use super::block::VisionBlock;
use super::merger::PatchMerger;
use super::patch_embed::PatchEmbed;
use super::rotary::{
    compute_vision_position_ids, compute_vision_rotary_emb, VisionRotaryEmbedding,
};

/// Complete CogViT vision encoder.
///
/// Pipeline:
/// 1. Patch embedding (Conv3d → Linear)
/// 2. 24 transformer blocks with 2D RoPE
/// 3. Post-layernorm
/// 4. Spatial 2×2 downsample (via Conv2d or reshape)
/// 5. Patch merger (SwiGLU projection to text decoder dim)
pub struct VisionEncoder {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    post_layernorm: RmsNorm,
    rotary_emb: VisionRotaryEmbedding,
    downsample: Conv2d,
    merger: PatchMerger,
    spatial_merge_size: usize,
    hidden_size: usize,
    out_hidden_size: usize,
}

impl VisionEncoder {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(config, vb.pp("patch_embed"))?;

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(VisionBlock::new(config, vb.pp(format!("blocks.{i}")))?);
        }

        let post_layernorm =
            rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("post_layernorm"))?;

        let head_dim = config.head_dim();
        // Python uses dim = head_dim // 2 for VisionRotaryEmbedding
        let rotary_emb = VisionRotaryEmbedding::new(head_dim / 2, 10000.0, vb.device())?;

        // Downsample: Conv2d(hidden_size, out_hidden_size, kernel=merge, stride=merge)
        let downsample_config = Conv2dConfig {
            stride: config.spatial_merge_size,
            ..Default::default()
        };
        let downsample = conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            downsample_config,
            vb.pp("downsample"),
        )?;

        let merger = PatchMerger::new(config, vb.pp("merger"))?;

        Ok(Self {
            patch_embed,
            blocks,
            post_layernorm,
            rotary_emb,
            downsample,
            merger,
            spatial_merge_size: config.spatial_merge_size,
            hidden_size: config.hidden_size,
            out_hidden_size: config.out_hidden_size,
        })
    }

    /// Forward pass.
    ///
    /// - `pixel_values`: [num_patches, patch_volume] — preprocessed image patches
    /// - `grid_thw`: [temporal, grid_h, grid_w]
    ///
    /// Returns: merged image embeddings [num_merged_patches, out_hidden_size]
    /// where num_merged_patches = num_patches / (spatial_merge_size^2)
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: [u32; 3],
    ) -> Result<Tensor> {
        let device = pixel_values.device();

        // 1. Patch embedding
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // 2. Compute rotary position embeddings
        let position_ids =
            compute_vision_position_ids(grid_thw, self.spatial_merge_size, device)?;
        let max_grid = grid_thw[1].max(grid_thw[2]);
        let (cos, sin) = compute_vision_rotary_emb(
            &position_ids,
            &self.rotary_emb,
            max_grid,
            device,
        )?;

        // 3. Run through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cos, &sin)?;
        }

        // 4. Post-layernorm
        hidden_states = self.post_layernorm.forward(&hidden_states)?;

        // 5. Spatial downsample via Conv2d.
        //
        // Patches are in merge-grouped order: consecutive groups of merge*merge
        // patches form a spatial 2x2 block. This matches the Python reference:
        //   hidden_states.view(-1, merge, merge, C).permute(0, 3, 1, 2)
        //   → [N/4, C, merge, merge] → Conv2d → [N/4, out_C, 1, 1]
        let merge = self.spatial_merge_size;
        let n = hidden_states.dim(0)?;
        let n_merged = n / (merge * merge);

        // [N, C] → [N/4, merge, merge, C] → permute → [N/4, C, merge, merge]
        let hidden_states = hidden_states.reshape((n_merged, merge, merge, self.hidden_size))?;
        let hidden_states = hidden_states.permute((0, 3, 1, 2))?.contiguous()?;

        // Conv2d(kernel=merge, stride=merge): [N/4, C, 2, 2] → [N/4, out_C, 1, 1]
        let hidden_states = self.downsample.forward(&hidden_states)?;

        // Flatten: [N/4, out_C, 1, 1] → [N/4, out_C]
        let hidden_states = hidden_states.reshape((n_merged, self.out_hidden_size))?;

        // 6. Patch merger (SwiGLU projection)
        let merged = self.merger.forward(&hidden_states)?;

        Ok(merged)
    }
}
