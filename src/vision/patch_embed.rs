use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::VisionConfig;

/// Patch embedding implemented as a Linear projection.
///
/// The original PyTorch uses Conv3d(in_channels, hidden_size, kernel=[T,H,W], stride=[T,H,W]).
/// Since kernel == stride (no overlap), each patch is independent, so this is equivalent to
/// a Linear projection from flattened patch to hidden_size.
///
/// Input: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
/// Output: [num_patches, hidden_size]
pub struct PatchEmbed {
    proj: Linear,
}

impl PatchEmbed {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_features =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
        let out_features = config.hidden_size;

        // The Conv3d weight has shape [out_channels, in_channels, T, H, W] = [1024, 3, 2, 14, 14].
        // We load it with the original shape then reshape to [out_features, in_features] for Linear.
        let weight = vb.get(
            (out_features, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size),
            "proj.weight",
        )?;
        let weight = weight.reshape((out_features, in_features))?;
        let bias = vb.get(out_features, "proj.bias")?;
        let proj = Linear::new(weight, Some(bias));

        Ok(Self { proj })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.proj.forward(xs)
    }
}
