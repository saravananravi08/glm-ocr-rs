use candle_core::{Result, Tensor};
use candle_nn::{linear_b, Linear, Module, VarBuilder};

use crate::config::VisionConfig;

/// SwiGLU MLP for vision encoder.
///
/// gate_proj + up_proj → SiLU(gate) * up → down_proj
/// All linear layers have bias=true (unlike text decoder).
pub struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMlp {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let bias = config.attention_bias; // vision uses bias=true

        let gate_proj = linear_b(hidden, intermediate, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden, intermediate, bias, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate, hidden, bias, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for VisionMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
