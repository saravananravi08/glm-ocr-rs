//! Q8_0 quantization utilities for GLM-OCR inference.
//!
//! Quantizes Linear layer weights from F32 to Q8_0 at load time,
//! reducing memory bandwidth by ~4x for memory-bound decode steps.

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
    /// Loads the weight as F32 from safetensors, then quantizes to Q8_0.
    /// No bias support (GLM-OCR uses linear_no_bias throughout).
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let qtensor = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
        let inner = QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}
