use candle_core::quantized::GgmlDType;
use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder};

use crate::config::TextConfig;
use crate::quantize::QLinear;
use super::rotary::apply_rotary_pos_emb;

/// KV-cache for a single decoder layer.
pub struct KvCache {
    pub key: Tensor,   // [batch, num_kv_heads, cached_len, head_dim]
    pub value: Tensor, // [batch, num_kv_heads, cached_len, head_dim]
}

/// Grouped Query Attention (GQA) with KV-cache support.
///
/// 16 query heads, 8 KV heads (groups=2).
/// Q projection: hidden→num_heads*head_dim = 1536→2048
/// K projection: hidden→num_kv_heads*head_dim = 1536→1024
/// V projection: hidden→num_kv_heads*head_dim = 1536→1024
/// O projection: num_heads*head_dim→hidden = 2048→1536
pub struct TextAttention {
    q_proj: Box<dyn Module>,
    k_proj: Box<dyn Module>,
    v_proj: Box<dyn Module>,
    o_proj: Box<dyn Module>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scale: f64,
}

impl TextAttention {
    pub fn new(config: &TextConfig, vb: VarBuilder, qdtype: Option<GgmlDType>) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let (q_proj, k_proj, v_proj, o_proj): (
            Box<dyn Module>,
            Box<dyn Module>,
            Box<dyn Module>,
            Box<dyn Module>,
        ) = if let Some(qdt) = qdtype {
            (
                Box::new(QLinear::new(hidden, num_heads * head_dim, vb.pp("q_proj"), qdt)?),
                Box::new(QLinear::new(hidden, num_kv_heads * head_dim, vb.pp("k_proj"), qdt)?),
                Box::new(QLinear::new(hidden, num_kv_heads * head_dim, vb.pp("v_proj"), qdt)?),
                Box::new(QLinear::new(num_heads * head_dim, hidden, vb.pp("o_proj"), qdt)?),
            )
        } else {
            (
                Box::new(linear_no_bias(hidden, num_heads * head_dim, vb.pp("q_proj"))?),
                Box::new(linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?),
                Box::new(linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?),
                Box::new(linear_no_bias(num_heads * head_dim, hidden, vb.pp("o_proj"))?),
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Forward pass with optional KV-cache.
    ///
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `cos`, `sin`: rotary embeddings [batch, seq_len, head_dim]
    /// - `attention_mask`: optional causal mask [batch, 1, seq_len, total_len]
    /// - `cache`: optional KV-cache from previous steps
    ///
    /// Returns: (output [batch, seq_len, hidden], updated KvCache)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        cache: Option<KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [batch, seq, num_heads*head_dim] → [batch, num_heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply rotary embeddings
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // Update KV-cache
        let (k, v) = if let Some(prev_cache) = cache {
            let k = Tensor::cat(&[&prev_cache.key, &k], 2)?;
            let v = Tensor::cat(&[&prev_cache.value, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        let new_cache = KvCache {
            key: k.clone(),
            value: v.clone(),
        };

        // Expand KV heads for GQA: repeat each KV head for `num_kv_groups` query heads
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Attention: softmax((Q @ K^T) * scale + mask) @ V
        // CUDA matmul requires contiguous tensors after transpose
        let attn_weights = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.contiguous()?.matmul(&v.contiguous()?)?;

        // Reshape: [batch, num_heads, seq, head_dim] → [batch, seq, num_heads*head_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        let output = self.o_proj.forward(&attn_output)?;
        Ok((output, new_cache))
    }

    /// Repeat KV heads to match query head count for GQA.
    /// [batch, num_kv_heads, seq, head_dim] → [batch, num_heads, seq, head_dim]
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        // Expand: [batch, kv_heads, 1, seq, head_dim] → [batch, kv_heads, groups, seq, head_dim]
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim))?
            .reshape((batch, self.num_heads, seq_len, head_dim))?;
        Ok(x)
    }
}
