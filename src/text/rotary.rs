use candle_core::{DType, Device, IndexOp, Result, Tensor};

use crate::config::MROPE_SECTIONS;

/// 3D Multi-dimensional Rotary Position Embedding (mRoPE) for the text decoder.
///
/// Position IDs are 3-dimensional: [temporal, height, width].
/// The head_dim is split into sections [16, 24, 24] (sums to 64 = head_dim/2).
/// Each section uses one position dimension.
pub struct TextRotaryEmbedding {
    inv_freq: Tensor,
    head_dim: usize,
}

impl TextRotaryEmbedding {
    pub fn new(head_dim: usize, rope_theta: f64, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq_data = vec![0f32; half_dim];
        for i in 0..half_dim {
            inv_freq_data[i] = 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32;
        }
        let inv_freq = Tensor::from_vec(inv_freq_data, half_dim, device)?;
        Ok(Self {
            inv_freq,
            head_dim,
        })
    }

    /// Compute (cos, sin) from 3D position IDs.
    ///
    /// `position_ids`: [3, batch, seq_len] — temporal, height, width positions
    ///
    /// Returns (cos, sin) each of shape [batch, seq_len, head_dim]
    pub fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // Expand inv_freq: [1, 1, head_dim/2, 1] for broadcasting with position_ids
        let inv_freq = self
            .inv_freq
            .reshape((1, 1, self.head_dim / 2, 1))?
            .to_dtype(DType::F32)?;

        // inv_freq_expanded: [3, batch, head_dim/2, 1]
        let batch = position_ids.dim(1)?;
        let inv_freq_expanded = inv_freq.expand((3, batch, self.head_dim / 2, 1))?;

        // position_ids_expanded: [3, batch, 1, seq_len]
        let position_ids_f = position_ids.to_dtype(DType::F32)?.unsqueeze(2)?;

        // freqs: [3, batch, head_dim/2, seq_len] → transpose → [3, batch, seq_len, head_dim/2]
        let freqs = inv_freq_expanded
            .contiguous()?
            .matmul(&position_ids_f.contiguous()?)?
            .transpose(2, 3)?;

        // Apply mRoPE: split head_dim/2 into sections, pick from corresponding position dim
        let freqs = self.apply_mrope(&freqs)?; // [batch, seq_len, head_dim/2]

        // Duplicate for full head_dim: [batch, seq_len, head_dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 2)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok((cos, sin))
    }

    /// Apply multi-dimensional RoPE by selecting from the 3 position dimensions.
    ///
    /// freqs: [3, batch, seq_len, head_dim/2]
    /// sections: [16, 24, 24] — how many freq entries to take from each dim
    ///
    /// Returns: [batch, seq_len, head_dim/2]
    fn apply_mrope(&self, freqs: &Tensor) -> Result<Tensor> {
        let sections = MROPE_SECTIONS;
        let mut chunks = Vec::new();
        let mut offset = 0;

        for (i, &section_size) in sections.iter().enumerate() {
            // Take `section_size` entries from dimension i % 3
            let dim_idx = i % 3;
            let chunk = freqs
                .i(dim_idx)?                              // [batch, seq_len, head_dim/2]
                .narrow(2, offset, section_size)?;         // [batch, seq_len, section_size]
            chunks.push(chunk);
            offset += section_size;
        }

        Tensor::cat(&chunks, 2)
    }
}

/// Apply rotary embeddings to Q and K using the interleaved method (rotate_half_llm).
///
/// This is different from the vision encoder's rotate_half!
/// GLM-OCR text decoder interleaves even/odd indices instead of splitting halves.
///
/// q, k: [batch, num_heads, seq_len, head_dim]
/// cos, sin: [batch, seq_len, head_dim]
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Unsqueeze for head dimension: [batch, 1, seq_len, head_dim]
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    // For interleaved RoPE, cos/sin need to be processed:
    // Take first half and repeat_interleave(2) along last dim
    let half_dim = cos.dim(3)? / 2;
    let cos_half = cos.narrow(3, 0, half_dim)?;
    let sin_half = sin.narrow(3, 0, half_dim)?;

    // repeat_interleave(2): [a, b, c] → [a, a, b, b, c, c]
    let cos_interleaved = repeat_interleave_2(&cos_half)?;
    let sin_interleaved = repeat_interleave_2(&sin_half)?;

    let rotary_dim = cos_interleaved.dim(3)?;

    // Split q, k into rotary part and passthrough part
    let q_rot = q.narrow(3, 0, rotary_dim)?;
    let q_pass = q.narrow(3, rotary_dim, q.dim(3)? - rotary_dim)?;
    let k_rot = k.narrow(3, 0, rotary_dim)?;
    let k_pass = k.narrow(3, rotary_dim, k.dim(3)? - rotary_dim)?;

    // Apply interleaved rotation (broadcast_mul for [batch, heads, seq, dim] * [batch, 1, seq, dim])
    let q_embed =
        q_rot.to_dtype(DType::F32)?.broadcast_mul(&cos_interleaved)?
            .add(&rotate_half_llm(&q_rot)?.broadcast_mul(&sin_interleaved)?)?;
    let k_embed =
        k_rot.to_dtype(DType::F32)?.broadcast_mul(&cos_interleaved)?
            .add(&rotate_half_llm(&k_rot)?.broadcast_mul(&sin_interleaved)?)?;

    // Concatenate back
    let q_out = Tensor::cat(&[&q_embed.to_dtype(q.dtype())?, &q_pass], 3)?;
    let k_out = Tensor::cat(&[&k_embed.to_dtype(k.dtype())?, &k_pass], 3)?;

    Ok((q_out, k_out))
}

/// Interleaved rotate_half (GLM-OCR text decoder variant).
///
/// Takes even indices as x1, odd indices as x2, returns [-x2, x1] interleaved.
fn rotate_half_llm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let dims = x.dims().to_vec();
    let ndim = dims.len();
    let last = ndim - 1;
    let dim_size = dims[last];
    let half = dim_size / 2;

    // Reshape [..., dim] → [..., dim/2, 2] to separate even/odd
    let mut paired_dims = dims[..last].to_vec();
    paired_dims.push(half);
    paired_dims.push(2);
    let x_paired = x.reshape(&*paired_dims)?;

    // Select even (index 0) and odd (index 1) from the LAST dim using narrow
    let x1 = x_paired.narrow(ndim, 0, 1)?.squeeze(ndim)?; // [..., dim/2] even indices
    let x2 = x_paired.narrow(ndim, 1, 1)?.squeeze(ndim)?; // [..., dim/2] odd indices

    // Interleave [-x2, x1]: stack along a new last dim then flatten
    let neg_x2 = x2.neg()?;
    let stacked = Tensor::stack(&[&neg_x2, &x1], ndim)?; // [..., dim/2, 2]

    // Flatten last two dims back to original shape
    let mut out_dims = dims[..last].to_vec();
    out_dims.push(dim_size);
    stacked.reshape(&*out_dims)
}

/// Repeat each element along the last dimension 2 times.
/// [a, b, c] → [a, a, b, b, c, c]
fn repeat_interleave_2(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let last = dims.len() - 1;

    // Unsqueeze last dim, expand, reshape
    let expanded = x.unsqueeze(last + 1)?; // [..., dim, 1]
    let expanded = expanded.expand(
        dims.iter()
            .chain(std::iter::once(&2))
            .copied()
            .collect::<Vec<_>>(),
    )?;

    let mut new_dims = dims[..last].to_vec();
    new_dims.push(dims[last] * 2);
    expanded.reshape(&*new_dims)
}
