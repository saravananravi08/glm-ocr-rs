use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

use crate::config::GlmOcrConfig;

const DEFAULT_MODEL_ID: &str = "unsloth/GLM-OCR";

pub struct ModelLoader {
    model_id: String,
    cache_dir: Option<PathBuf>,
}

impl ModelLoader {
    pub fn new(model_id: Option<&str>) -> Self {
        Self {
            model_id: model_id.unwrap_or(DEFAULT_MODEL_ID).to_string(),
            cache_dir: None,
        }
    }

    pub fn with_cache_dir(mut self, dir: PathBuf) -> Self {
        self.cache_dir = Some(dir);
        self
    }

    /// Download and load the model config from HuggingFace.
    pub fn load_config(&self) -> Result<GlmOcrConfig> {
        let config_path = self.get_file("config.json")?;
        let config_str =
            std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
        let config: GlmOcrConfig =
            serde_json::from_str(&config_str).context("Failed to parse config.json")?;
        Ok(config)
    }

    /// Download and load model weights as a VarBuilder.
    pub fn load_weights(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let weights_path = self.get_file("model.safetensors")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .context("Failed to load model weights")?
        };
        Ok(vb)
    }

    /// Get path to the tokenizer.json file.
    pub fn tokenizer_path(&self) -> Result<PathBuf> {
        self.get_file("tokenizer.json")
    }

    /// Download a file from HuggingFace hub and return its local path.
    fn get_file(&self, filename: &str) -> Result<PathBuf> {
        let api = Api::new().context("Failed to create HF API")?;

        let repo = api.repo(Repo::new(self.model_id.clone(), RepoType::Model));
        let path = repo
            .get(filename)
            .with_context(|| format!("Failed to download {filename} from {}", self.model_id))?;
        Ok(path)
    }
}
