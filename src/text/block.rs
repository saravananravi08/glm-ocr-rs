use candle_core::quantized::GgmlDType;
use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

use crate::config::TextConfig;
use crate::nn_utils::rms_norm;
use super::attention::{KvCache, TextAttention};
use super::mlp::TextMlp;

/// Text decoder layer with 4 RMSNorm layers (GLM-OCR specific).
///
/// Forward pass:
///   residual = x
///   x = input_layernorm(x)
///   x = self_attn(x)
///   x = post_self_attn_layernorm(x)
///   x = residual + x
///
///   residual = x
///   x = post_attention_layernorm(x)
///   x = mlp(x)
///   x = post_mlp_layernorm(x)
///   x = residual + x
pub struct TextDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMlp,
    input_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
}

impl TextDecoderLayer {
    pub fn new(config: &TextConfig, vb: VarBuilder, qdtype: Option<GgmlDType>) -> Result<Self> {
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;

        let self_attn = TextAttention::new(config, vb.pp("self_attn"), qdtype)?;
        let mlp = TextMlp::new(config, vb.pp("mlp"), qdtype)?;
        let input_layernorm = rms_norm(hidden, eps, vb.pp("input_layernorm"))?;
        let post_self_attn_layernorm =
            rms_norm(hidden, eps, vb.pp("post_self_attn_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(hidden, eps, vb.pp("post_attention_layernorm"))?;
        let post_mlp_layernorm = rms_norm(hidden, eps, vb.pp("post_mlp_layernorm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_self_attn_layernorm,
            post_attention_layernorm,
            post_mlp_layernorm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        cache: Option<KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        // Self-attention block
        let residual = hidden_states;
        let h = self.input_layernorm.forward(hidden_states)?;
        let (h, new_cache) = self.self_attn.forward(&h, cos, sin, attention_mask, cache)?;
        let h = self.post_self_attn_layernorm.forward(&h)?;
        let hidden_states = (residual + h)?;

        // MLP block
        let residual = &hidden_states;
        let h = self.post_attention_layernorm.forward(&hidden_states)?;
        let h = candle_nn::Module::forward(&self.mlp, &h)?;
        let h = self.post_mlp_layernorm.forward(&h)?;
        let hidden_states = (residual + h)?;

        Ok((hidden_states, new_cache))
    }
}
