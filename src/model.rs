use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::config::GlmOcrConfig;
use crate::text::TextDecoder;
use crate::vision::VisionEncoder;

/// Top-level GLM-OCR model combining vision encoder and text decoder.
pub struct GlmOcrModel {
    pub vision_encoder: VisionEncoder,
    pub text_decoder: TextDecoder,
    pub config: GlmOcrConfig,
    pub device: Device,
    pub dtype: DType,
}

impl GlmOcrModel {
    pub fn new(config: &GlmOcrConfig, vb: VarBuilder, device: &Device, dtype: DType, qdtype: Option<GgmlDType>) -> candle_core::Result<Self> {
        let model_vb = vb.pp("model");
        let vision_encoder = VisionEncoder::new(&config.vision_config, model_vb.pp("visual"))?;
        let text_decoder = TextDecoder::new(
            &config.text_config,
            model_vb.pp("language_model"),
            vb.pp("lm_head"),
            qdtype,
        )?;

        Ok(Self {
            vision_encoder,
            text_decoder,
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Embed text tokens and replace image placeholder positions with vision embeddings.
    ///
    /// `input_ids`: [batch, seq_len] — contains image_token_id at placeholder positions
    /// `image_embeds`: [num_image_tokens, hidden_size] — vision encoder output
    ///
    /// Returns: [batch, seq_len, hidden_size]
    pub fn embed_and_merge(
        &self,
        input_ids: &[u32],
        image_embeds: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.len();
        let ids_tensor = Tensor::from_vec(
            input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (1, seq_len),
            &self.device,
        )?;

        // Embed all tokens: [1, seq, hidden]
        let text_embeds = self.text_decoder.embed(&ids_tensor)?;
        let text_embeds = text_embeds.squeeze(0)?; // [seq, hidden]

        // Build merged embeddings by collecting segments
        let image_token_id = self.config.image_token_id;
        let mut segments: Vec<Tensor> = Vec::new();
        let mut img_idx: usize = 0;
        let mut text_start: usize = 0;

        for (pos, &token_id) in input_ids.iter().enumerate() {
            if token_id == image_token_id {
                // Flush preceding text tokens as a segment
                if pos > text_start {
                    segments.push(text_embeds.narrow(0, text_start, pos - text_start)?);
                }
                // Add the vision embedding for this position
                segments.push(image_embeds.i(img_idx)?.unsqueeze(0)?); // [1, hidden]
                img_idx += 1;
                text_start = pos + 1;
            }
        }

        // Flush remaining text tokens
        if text_start < seq_len {
            segments.push(text_embeds.narrow(0, text_start, seq_len - text_start)?);
        }

        // Concatenate all segments: [seq, hidden] → unsqueeze → [1, seq, hidden]
        let merged = Tensor::cat(&segments, 0)?;
        Ok(merged.unsqueeze(0)?)
    }

    /// Compute 3D position IDs for the merged sequence.
    ///
    /// Follows the HuggingFace `get_rope_index` algorithm:
    /// - Groups consecutive tokens by modality (text=0, image=1)
    /// - Text tokens: all 3 dims get the same sequential position
    /// - Image tokens: temporal=start_pos, height/width=2D grid + start_pos
    /// - After image block: current_pos += max(grid_h, grid_w) / merge_size
    ///
    /// Returns: ([3, 1, seq_len] position tensor, next_position for decode)
    pub fn compute_3d_positions(
        &self,
        input_ids: &[u32],
        grid_thw: [u32; 3],
    ) -> candle_core::Result<(Tensor, i64)> {
        let seq_len = input_ids.len();
        let merge = self.config.vision_config.spatial_merge_size as u32;
        let image_token_id = self.config.image_token_id;

        // Build mm_token_type_ids: 0=text, 1=image
        let token_types: Vec<u8> = input_ids
            .iter()
            .map(|&id| if id == image_token_id { 1 } else { 0 })
            .collect();

        // Group consecutive tokens by modality type
        // Each group: (modality_type, start_idx, end_idx)
        let mut groups: Vec<(u8, usize, usize)> = Vec::new();
        if !token_types.is_empty() {
            let mut current_type = token_types[0];
            let mut start = 0;
            for i in 1..token_types.len() {
                if token_types[i] != current_type {
                    groups.push((current_type, start, i));
                    current_type = token_types[i];
                    start = i;
                }
            }
            groups.push((current_type, start, token_types.len()));
        }

        let mut temporal_pos = vec![0i64; seq_len];
        let mut height_pos = vec![0i64; seq_len];
        let mut width_pos = vec![0i64; seq_len];

        let mut current_pos: i64 = 0;

        for (modality_type, start_idx, end_idx) in &groups {
            if *modality_type == 0 {
                // Text tokens: all 3 dims get same sequential position
                let text_len = end_idx - start_idx;
                for i in 0..text_len {
                    let pos = current_pos + i as i64;
                    temporal_pos[start_idx + i] = pos;
                    height_pos[start_idx + i] = pos;
                    width_pos[start_idx + i] = pos;
                }
                current_pos += text_len as i64;
            } else {
                // Image tokens: 3D positions from get_vision_position_ids
                let [t, grid_h, grid_w] = grid_thw;
                let temp_merge_size = t; // GLM-OCR: temp_merge_size = grid_thw[0]
                let llm_grid_t = t / temp_merge_size;
                let llm_grid_h = grid_h / merge;
                let llm_grid_w = grid_w / merge;

                // Generate vision position IDs matching Python reference:
                // width: arange(start, start+w).repeat(h*t)
                // height: arange(start, start+h).repeat_interleave(w*t)
                // temporal: full(seq_len, start * time_interval) [time_interval=1]
                let start = current_pos;
                let mut idx = 0;
                for _ft in 0..llm_grid_t {
                    for fh in 0..llm_grid_h {
                        for fw in 0..llm_grid_w {
                            let pos = start_idx + idx as usize;
                            if pos < seq_len {
                                temporal_pos[pos] = start; // constant for images
                                height_pos[pos] = start + fh as i64;
                                width_pos[pos] = start + fw as i64;
                            }
                            idx += 1;
                        }
                    }
                }

                // Advance current_pos by max(H, W) / merge_size (NOT by token count!)
                current_pos += (grid_h.max(grid_w) / merge) as i64;
            }
        }

        // Stack into [3, 1, seq_len]
        let t = Tensor::from_vec(temporal_pos, (1, seq_len), &self.device)?;
        let h = Tensor::from_vec(height_pos, (1, seq_len), &self.device)?;
        let w = Tensor::from_vec(width_pos, (1, seq_len), &self.device)?;

        // next_position for autoregressive decode: current_pos is already set correctly
        Ok((Tensor::stack(&[&t, &h, &w], 0)?, current_pos))
    }
}

