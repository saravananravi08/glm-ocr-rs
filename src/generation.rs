use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::image_processor::preprocess_image;
use crate::model::GlmOcrModel;
use crate::text::attention::KvCache;
use crate::tokenizer::GlmOcrTokenizer;
use image::DynamicImage;

/// Generate text from an image using GLM-OCR.
///
/// This function orchestrates the full pipeline:
/// 1. Preprocess image
/// 2. Run vision encoder
/// 3. Build token sequence with image placeholders
/// 4. Embed + merge vision tokens
/// 5. Prefill the decoder
/// 6. Autoregressively generate tokens until EOS or max_tokens
pub fn generate(
    model: &GlmOcrModel,
    tokenizer: &GlmOcrTokenizer,
    image: &DynamicImage,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    let device = &model.device;

    // 1. Preprocess image and cast to model dtype
    let (pixel_values, grid_thw) = preprocess_image(image, &model.config.vision_config, device)?;
    let pixel_values = pixel_values.to_dtype(model.dtype)?;

    // 2. Run vision encoder (single forward pass)
    let image_embeds = model.vision_encoder.forward(&pixel_values, grid_thw)?;
    let num_image_tokens = image_embeds.dim(0)?;

    // 3. Build input token sequence
    let input_ids = tokenizer.build_input_ids(prompt, num_image_tokens)?;
    tracing::debug!("grid_thw: {:?}, num_image_tokens: {}, input_ids len: {}", grid_thw, num_image_tokens, input_ids.len());

    // 4. Embed tokens + merge vision embeddings
    let inputs_embeds = model.embed_and_merge(&input_ids, &image_embeds)?;

    // 5. Compute 3D position IDs (returns next_position for decode)
    let (position_ids, next_decode_pos) = model.compute_3d_positions(&input_ids, grid_thw)?;
    tracing::debug!("next_decode_pos: {}", next_decode_pos);

    // 6. Build causal attention mask for prefill (must match model dtype for GPU)
    let seq_len = input_ids.len();
    let attention_mask = create_causal_mask(seq_len, device)?.to_dtype(model.dtype)?;

    // 7. Prefill: forward pass on entire sequence
    let kv_caches: Vec<Option<KvCache>> = Vec::new();
    let (logits, mut kv_caches) = model.text_decoder.forward_with_cache(
        &inputs_embeds,
        &position_ids,
        Some(&attention_mask),
        kv_caches,
    )?;

    // 8. Get first generated token (greedy argmax, no penalty for first token)
    let mut generated_tokens: Vec<u32> = Vec::new();
    let next_token = argmax_with_debug(&logits, "prefill")?;
    let mut next_token = next_token;
    tracing::debug!("First token: {} (eos={})", next_token, tokenizer.is_eos(next_token));
    let mut output_tokens = Vec::new();

    // 9. Autoregressive decode loop
    // Position for decode uses the position counter from get_rope_index, NOT seq_len
    let mut current_pos = next_decode_pos;
    tracing::debug!("Decode start pos: {}, seq_len: {}, num_image_tokens: {}", current_pos, seq_len, num_image_tokens);
    for step in 0..max_tokens {
        if tokenizer.is_eos(next_token) {
            tracing::debug!("EOS at step {}, token {}", step, next_token);
            break;
        }
        output_tokens.push(next_token);
        generated_tokens.push(next_token);
        if step < 30 {
            tracing::debug!("Step {}: token {} pos={}", step, next_token, current_pos);
        }

        // Embed single token
        let token_tensor = Tensor::from_vec(
            vec![next_token as i64],
            (1, 1),
            device,
        )?;
        let token_embeds = model.text_decoder.embed(&token_tensor)?; // [1, 1, hidden]

        // Position IDs for this step: all 3 dims get same sequential value
        let pos_val = current_pos as i64;
        let pos = Tensor::from_vec(vec![pos_val; 3], (3, 1, 1), device)?;

        // No attention mask needed for single-token decode with KV-cache
        let (logits, new_caches) = model.text_decoder.forward_with_cache(
            &token_embeds,
            &pos,
            None,
            kv_caches,
        )?;
        kv_caches = new_caches;

        next_token = argmax_with_penalty(&logits, &generated_tokens, 1.0)?;
        current_pos += 1;
    }

    // 10. Decode tokens to text
    let text = tokenizer.decode(&output_tokens, true)?;
    Ok(text)
}

/// Create a causal attention mask.
/// Returns a [1, 1, seq_len, seq_len] mask where future positions are -inf.
fn create_causal_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
}

/// Get argmax with debug logging of top-5 tokens.
fn argmax_with_debug(logits: &Tensor, label: &str) -> candle_core::Result<u32> {
    let logits_flat = logits.squeeze(0)?.squeeze(0)?;
    let logits_vec = logits_flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // Get top 5
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top5: Vec<_> = indexed.iter().take(5).collect();
    tracing::debug!("[{}] Top-5 logits: {:?}", label, top5);

    Ok(indexed[0].0 as u32)
}

/// Get the argmax of the last token's logits with repetition penalty.
fn argmax_with_penalty(
    logits: &Tensor,
    generated: &[u32],
    penalty: f32,
) -> candle_core::Result<u32> {
    // logits: [batch=1, 1, vocab_size]
    let logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]
    let mut logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // Apply repetition penalty to previously generated tokens
    for &token_id in generated {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            if logits_vec[idx] > 0.0 {
                logits_vec[idx] /= penalty;
            } else {
                logits_vec[idx] *= penalty;
            }
        }
    }

    // Argmax
    let (best_idx, _) = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    Ok(best_idx as u32)
}
