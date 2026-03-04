use candle_core::Result;
use candle_nn::{RmsNorm, VarBuilder};

/// Load an RmsNorm layer from a VarBuilder.
/// Candle 0.8's RmsNorm::new takes (weight: Tensor, eps: f64).
pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}
