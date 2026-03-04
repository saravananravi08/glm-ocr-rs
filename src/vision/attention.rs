use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{linear_b, Linear, Module, RmsNorm, VarBuilder};

use crate::config::VisionConfig;
use crate::nn_utils::rms_norm;
use super::rotary::apply_rotary_pos_emb_vision;

/// Multi-head attention for the vision encoder.
///
/// Features:
/// - Fused QKV projection
/// - QK-normalization (RMSNorm per head dimension)
/// - 2D rotary position embeddings
/// - Bidirectional attention (no causal mask)
pub struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim();
        let bias = config.attention_bias;

        let qkv = linear_b(hidden, 3 * hidden, bias, vb.pp("qkv"))?;
        let proj = linear_b(hidden, hidden, bias, vb.pp("proj"))?;
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: [seq_len, hidden_size]
    /// - `cos`, `sin`: rotary embeddings [seq_len, head_dim]
    ///
    /// Returns: [seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(0)?;

        // QKV projection: [seq, hidden] -> [seq, 3*hidden]
        let qkv = self.qkv.forward(hidden_states)?;

        // Reshape to [seq, 3, num_heads, head_dim]
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;

        // Split into Q, K, V: each [seq, num_heads, head_dim]
        let q = qkv.i((.., 0, .., ..))?.contiguous()?;
        let k = qkv.i((.., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., 2, .., ..))?.contiguous()?;

        // QK-norm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply rotary embeddings
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        // Transpose for attention: [num_heads, seq, head_dim]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        // Attention: softmax(Q @ K^T * scale) @ V
        // CUDA matmul requires contiguous tensors after transpose
        let attn_weights = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.contiguous()?.matmul(&v)?;

        // Transpose back: [num_heads, seq, head_dim] -> [seq, num_heads, head_dim]
        let attn_output = attn_output.transpose(0, 1)?.contiguous()?;

        // Reshape: [seq, num_heads * head_dim] = [seq, hidden]
        let attn_output = attn_output.reshape((seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.proj.forward(&attn_output)
    }
}
