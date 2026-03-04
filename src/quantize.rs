//! Quantization utilities for GLM-OCR inference.
//!
//! Supports Q8_0 and Q4_0 quantization at load time.
//! Q8_0: ~4x memory bandwidth reduction, minimal quality loss.
//! Q4_0: ~8x memory bandwidth reduction, some quality loss but faster.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// A linear layer with quantized weights.
///
/// Wraps candle's QMatMul which performs on-the-fly block-wise
/// dequantization during matmul — weights are never fully dequantized.
pub struct QLinear {
    inner: QMatMul,
}

impl QLinear {
    /// Create a quantized linear layer from a VarBuilder.
    ///
    /// Loads the weight as F32 from safetensors, then quantizes to the given dtype.
    /// No bias support (GLM-OCR uses linear_no_bias throughout).
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder, qdtype: GgmlDType) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let qtensor = QTensor::quantize(&weight, qdtype)?;
        let inner = QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // QMatMul expects F32 input — cast if needed (e.g. F16 on GPU) and cast back
        let in_dtype = xs.dtype();
        if in_dtype == candle_core::DType::F32 {
            self.inner.forward(xs)
        } else {
            let out = self.inner.forward(&xs.to_dtype(candle_core::DType::F32)?)?;
            out.to_dtype(in_dtype)
        }
    }
}
