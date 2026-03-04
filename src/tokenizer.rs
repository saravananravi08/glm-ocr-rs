use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config::GlmOcrConfig;

/// Wrapper around HuggingFace tokenizer with GLM-OCR chat template support.
pub struct GlmOcrTokenizer {
    tokenizer: Tokenizer,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    eos_token_ids: Vec<u32>,
}

impl GlmOcrTokenizer {
    pub fn from_file(path: &Path, config: &GlmOcrConfig) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            tokenizer,
            image_token_id: config.image_token_id,
            image_start_token_id: config.image_start_token_id,
            image_end_token_id: config.image_end_token_id,
            eos_token_ids: config.text_config.eos_token_id.clone(),
        })
    }

    /// Build input token IDs for a single image OCR request.
    ///
    /// Matches the exact GLM-OCR chat template from HuggingFace:
    /// ```text
    /// [gMASK]<sop><|user|>\n
    /// <|begin_of_image|><|image|>...(N times)<|end_of_image|>
    /// {prompt}<|assistant|>\n
    /// ```
    pub fn build_input_ids(&self, prompt: &str, num_image_tokens: usize) -> Result<Vec<u32>> {
        // Encode the full template as a single string (with 1 image placeholder),
        // then expand the single image token to N tokens.
        let template = format!(
            "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>{prompt}<|assistant|>\n"
        );

        let encoding = self
            .tokenizer
            .encode(template.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let base_ids = encoding.get_ids();

        // Expand the single <|image|> token to num_image_tokens
        let mut ids = Vec::with_capacity(base_ids.len() + num_image_tokens - 1);
        for &token_id in base_ids {
            if token_id == self.image_token_id {
                for _ in 0..num_image_tokens {
                    ids.push(self.image_token_id);
                }
            } else {
                ids.push(token_id);
            }
        }

        Ok(ids)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
    }

    /// Check if a token ID is an end-of-sequence token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }
}
