use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder};

use crate::config::TextConfig;
use crate::quantize::QLinear;

/// SwiGLU MLP for text decoder.
///
/// Uses fused gate_up projection: Linear(hidden, 2*intermediate) then split.
/// No bias on any linear layer.
pub struct TextMlp {
    gate_up_proj: Box<dyn Module>,
    down_proj: Box<dyn Module>,
    intermediate_size: usize,
}

impl TextMlp {
    pub fn new(config: &TextConfig, vb: VarBuilder, quantize: bool) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        let (gate_up_proj, down_proj): (Box<dyn Module>, Box<dyn Module>) = if quantize {
            (
                Box::new(QLinear::new(hidden, 2 * intermediate, vb.pp("gate_up_proj"))?),
                Box::new(QLinear::new(intermediate, hidden, vb.pp("down_proj"))?),
            )
        } else {
            (
                Box::new(linear_no_bias(hidden, 2 * intermediate, vb.pp("gate_up_proj"))?),
                Box::new(linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?),
            )
        };

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: intermediate,
        })
    }
}

impl Module for TextMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(xs)?;

        // Split into gate and up: each [batch, seq, intermediate]
        let gate = up_states.narrow(2, 0, self.intermediate_size)?;
        let up = up_states.narrow(2, self.intermediate_size, self.intermediate_size)?;

        // SiLU(gate) * up
        let activated = candle_nn::Activation::Silu.forward(&gate)?;
        let hidden = (activated * up)?;

        self.down_proj.forward(&hidden)
    }
}
