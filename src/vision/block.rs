use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

use crate::config::VisionConfig;
use crate::nn_utils::rms_norm;
use super::attention::VisionAttention;
use super::mlp::VisionMlp;

/// Vision transformer block with pre-norm and residual connections.
///
/// x = x + attn(norm1(x))
/// x = x + mlp(norm2(x))
pub struct VisionBlock {
    norm1: RmsNorm,
    attn: VisionAttention,
    norm2: RmsNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm1"))?;
        let attn = VisionAttention::new(config, vb.pp("attn"))?;
        let norm2 = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm2"))?;
        let mlp = VisionMlp::new(config, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with pre-norm and residual
        let residual = hidden_states;
        let h = self.norm1.forward(hidden_states)?;
        let h = self.attn.forward(&h, cos, sin)?;
        let hidden_states = (residual + h)?;

        // MLP with pre-norm and residual
        let residual = &hidden_states;
        let h = self.norm2.forward(&hidden_states)?;
        let h = candle_nn::Module::forward(&self.mlp, &h)?;
        residual + h
    }
}
