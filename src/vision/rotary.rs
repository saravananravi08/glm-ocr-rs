use candle_core::{DType, Device, IndexOp, Result, Tensor};

/// 2D rotary position embeddings for vision patches.
///
/// Computes sin/cos embeddings from (height, width) position grids,
/// accounting for spatial_merge_size interleaving.
pub struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    pub fn new(dim: usize, theta: f64, device: &Device) -> Result<Self> {
        // inv_freq = 1.0 / (theta ^ (2i / dim)) for i in 0..dim/2
        let half_dim = dim / 2;
        let mut inv_freq_data = vec![0f32; half_dim];
        for i in 0..half_dim {
            inv_freq_data[i] = 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32;
        }
        let inv_freq = Tensor::from_vec(inv_freq_data, half_dim, device)?;
        Ok(Self { inv_freq })
    }

    /// Compute rotary embeddings for a sequence length.
    /// Returns freqs of shape [seqlen, dim/2].
    pub fn forward(&self, seqlen: u32, device: &Device) -> Result<Tensor> {
        let seq = Tensor::arange(0f32, seqlen as f32, device)?;
        // outer product: [seqlen] x [dim/2] -> [seqlen, dim/2]
        let freqs = seq.unsqueeze(1)?.contiguous()?.matmul(&self.inv_freq.unsqueeze(0)?.contiguous()?)?;
        Ok(freqs)
    }
}

/// Compute the 2D position IDs for vision patches in merge-grouped order.
///
/// Patches are ordered by merge blocks: for each block (by, bx) in the grid
/// of (grid_h/merge, grid_w/merge), for each sub-position (my, mx) within
/// the block. This matches the HuggingFace 9D reshape+transpose ordering.
///
/// Returns position_ids tensor of shape [num_patches, 2] with (h, w) for each patch.
pub fn compute_vision_position_ids(
    grid_thw: [u32; 3],
    spatial_merge_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let [t, h, w] = grid_thw;
    let merge = spatial_merge_size as u32;

    let mut pos_ids = Vec::new();

    for _frame in 0..t {
        let bh = h / merge;
        let bw = w / merge;

        for by in 0..bh {
            for bx in 0..bw {
                for my in 0..merge {
                    for mx in 0..merge {
                        let orig_y = by * merge + my;
                        let orig_x = bx * merge + mx;
                        pos_ids.push(orig_y as i64);
                        pos_ids.push(orig_x as i64);
                    }
                }
            }
        }
    }

    let total = (t * h * w) as usize;
    Tensor::from_vec(pos_ids, (total, 2), device)
}

/// Compute (cos, sin) rotary embeddings for vision attention.
///
/// Takes position_ids [num_patches, 2] and returns (cos, sin) each of shape [num_patches, head_dim].
///
/// Python reference:
///   rotary_emb dim = head_dim/2 = 32, so inv_freq has 16 elements
///   freqs[max_grid, 16], lookup h and w → [N, 16] each
///   cat([h_freqs, w_freqs], dim=-1) → [N, 32]
///   emb = cat([freqs, freqs], dim=-1) → [N, 64] = [N, head_dim]
///   cos, sin = emb.cos(), emb.sin()
pub fn compute_vision_rotary_emb(
    position_ids: &Tensor,
    rotary_emb: &VisionRotaryEmbedding,
    max_grid_size: u32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Compute base frequencies for max grid size
    // rotary_emb was created with dim = head_dim/2 = 32
    // inv_freq has dim/2 = 16 elements
    // freqs shape: [max_grid, 16]
    let freqs = rotary_emb.forward(max_grid_size, device)?;

    // Look up h and w frequencies separately
    let h_ids = position_ids.i((.., 0))?; // [num_patches]
    let w_ids = position_ids.i((.., 1))?; // [num_patches]

    let h_freqs = freqs.index_select(&h_ids.to_dtype(DType::U32)?, 0)?; // [N, 16]
    let w_freqs = freqs.index_select(&w_ids.to_dtype(DType::U32)?, 0)?; // [N, 16]

    // Concatenate h and w: [N, 16] + [N, 16] → [N, 32]
    let hw_freqs = Tensor::cat(&[&h_freqs, &w_freqs], 1)?;

    // Double to match head_dim: [N, 32] → [N, 64] = [N, head_dim]
    let full_freqs = Tensor::cat(&[&hw_freqs, &hw_freqs], 1)?;

    let cos = full_freqs.cos()?;
    let sin = full_freqs.sin()?;

    Ok((cos, sin))
}

/// Apply rotary position embeddings to query and key tensors (vision variant).
///
/// Uses the standard rotate_half (contiguous halves) method.
/// q, k: [seq_len, num_heads, head_dim]
/// cos, sin: [seq_len, head_dim]
pub fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Unsqueeze cos/sin to broadcast over heads: [seq_len, 1, head_dim]
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let q_embed = q.to_dtype(DType::F32)?.broadcast_mul(&cos)?.add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k.to_dtype(DType::F32)?.broadcast_mul(&cos)?.add(&rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed.to_dtype(q.dtype())?, k_embed.to_dtype(k.dtype())?))
}

/// Standard rotate_half: split last dim in contiguous halves, negate+swap.
/// [-x2, x1] where x1 = first half, x2 = second half.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let last_dim = x.dim(x.dims().len() - 1)?;
    let half = last_dim / 2;
    let x1 = x.narrow(x.dims().len() - 1, 0, half)?;
    let x2 = x.narrow(x.dims().len() - 1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], x.dims().len() - 1)
}
