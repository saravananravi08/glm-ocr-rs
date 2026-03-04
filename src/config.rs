use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct GlmOcrConfig {
    #[serde(default)]
    pub vision_config: VisionConfig,
    #[serde(default)]
    pub text_config: TextConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_image_start_token_id")]
    pub image_start_token_id: u32,
    #[serde(default = "default_image_end_token_id")]
    pub image_end_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    #[serde(default = "default_video_start_token_id")]
    pub video_start_token_id: u32,
    #[serde(default = "default_video_end_token_id")]
    pub video_end_token_id: u32,
}

fn default_image_token_id() -> u32 {
    59280
}
fn default_image_start_token_id() -> u32 {
    59256
}
fn default_image_end_token_id() -> u32 {
    59257
}
fn default_video_token_id() -> u32 {
    59281
}
fn default_video_start_token_id() -> u32 {
    59258
}
fn default_video_end_token_id() -> u32 {
    59259
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            depth: 24,
            num_heads: 16,
            patch_size: 14,
            image_size: 336,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            intermediate_size: 4096,
            out_hidden_size: 1536,
            rms_norm_eps: 1e-5,
            in_channels: 3,
            hidden_act: "silu".to_string(),
            attention_bias: true,
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

fn default_vision_hidden_size() -> usize {
    1024
}
fn default_depth() -> usize {
    24
}
fn default_num_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_image_size() -> usize {
    336
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_vision_intermediate_size() -> usize {
    4096
}
fn default_out_hidden_size() -> usize {
    1536
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_in_channels() -> usize {
    3
}
fn default_vision_hidden_act() -> String {
    "silu".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_nextn_predict_layers")]
    pub num_nextn_predict_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_eos_token_ids")]
    pub eos_token_id: Vec<u32>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 59392,
            hidden_size: 1536,
            num_hidden_layers: 16,
            num_nextn_predict_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 4608,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
            pad_token_id: 59246,
            eos_token_id: vec![59246, 59253],
        }
    }
}

impl TextConfig {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

fn default_vocab_size() -> usize {
    59392
}
fn default_text_hidden_size() -> usize {
    1536
}
fn default_num_hidden_layers() -> usize {
    16
}
fn default_num_nextn_predict_layers() -> usize {
    1
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_text_intermediate_size() -> usize {
    4608
}
fn default_head_dim() -> usize {
    128
}
fn default_text_hidden_act() -> String {
    "silu".to_string()
}
fn default_max_position_embeddings() -> usize {
    131072
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_pad_token_id() -> u32 {
    59246
}
fn default_eos_token_ids() -> Vec<u32> {
    vec![59246, 59253]
}

/// mRoPE section sizes — how the head_dim/2 is split across 3 position dimensions
/// [temporal, height, width] = [16, 24, 24] (sums to 64 = head_dim/2)
pub const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];

/// Image normalization constants (from preprocessor_config.json)
pub const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
