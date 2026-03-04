use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, LayerNorm, Linear, Module, VarBuilder};

use crate::config::VisionConfig;

/// Patch merger: takes downsampled vision features and applies SwiGLU projection.
///
/// Architecture (all dimensions are out_hidden_size=1536):
///   x = proj(x)                -- Linear 1536→1536 (no bias)
///   x = GELU(LayerNorm(x))    -- post_projection_norm + activation
///   x = down(SiLU(gate(x)) * up(x))  -- SwiGLU with intermediate=4608
///
/// Input: [N, out_hidden_size=1536] (after Conv2d downsample from 1024→1536)
/// Output: [N, out_hidden_size=1536]
pub struct PatchMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl PatchMerger {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        // After downsample Conv2d, the dim is already out_hidden_size (1536)
        let dim = config.out_hidden_size;
        // SwiGLU intermediate matches text decoder's intermediate_size (4608)
        let intermediate = 4608;

        let proj = linear_no_bias(dim, dim, vb.pp("proj"))?;
        let post_projection_norm = LayerNorm::new(
            vb.get(dim, "post_projection_norm.weight")?,
            vb.get(dim, "post_projection_norm.bias")?,
            config.rms_norm_eps,
        );
        let gate_proj = linear_no_bias(dim, intermediate, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(dim, intermediate, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate, dim, vb.pp("down_proj"))?;

        Ok(Self {
            proj,
            post_projection_norm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for PatchMerger {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(xs)?;
        let h = self.post_projection_norm.forward(&h)?;
        let h = candle_nn::Activation::Gelu.forward(&h)?;

        // SwiGLU
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(&h)?)?;
        let up = self.up_proj.forward(&h)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
