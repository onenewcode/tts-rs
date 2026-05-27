#![allow(dead_code)]

use std::env;
use std::path::PathBuf;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use tts_qwen::{
    CustomVoiceRequest, EngineConfig, SessionConfig, default_engine_config, default_session_config,
};

pub const SHORT_ASCII_TEXT: &str = "synthetic benchmark short ascii input";
pub const SHORT_ZH_TEXT: &str = "这是固定的合成基准短句。";
pub const MEDIUM_ZH_TEXT: &str = "这是固定的合成基准中等长度文本，用来稳定评估推理路径、tokenizer 编码和会话执行的耗时表现，不代表真实业务语料。";
pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Clone, Copy, Debug)]
pub struct SyntheticRequestCase {
    pub name: &'static str,
    pub text: &'static str,
    pub language: Option<&'static str>,
    pub speaker: Option<&'static str>,
}

impl SyntheticRequestCase {
    pub fn build_request(self) -> CustomVoiceRequest {
        CustomVoiceRequest {
            text: self.text.to_string(),
            language: self.language.map(str::to_string),
            speaker: self.speaker.map(str::to_string),
        }
    }
}

pub fn synthetic_request_cases() -> [SyntheticRequestCase; 3] {
    [
        SyntheticRequestCase {
            name: "short_ascii",
            text: SHORT_ASCII_TEXT,
            language: None,
            speaker: None,
        },
        SyntheticRequestCase {
            name: "short_zh",
            text: SHORT_ZH_TEXT,
            language: None,
            speaker: None,
        },
        SyntheticRequestCase {
            name: "medium_zh",
            text: MEDIUM_ZH_TEXT,
            language: None,
            speaker: None,
        },
    ]
}

pub fn require_model_dir(bench_name: &str) -> Option<PathBuf> {
    let Ok(model_dir) = env::var("QWEN_TTS_MODEL_DIR") else {
        eprintln!("skipping {bench_name}: QWEN_TTS_MODEL_DIR is not set");
        return None;
    };
    let path = PathBuf::from(model_dir);
    if !path.join("config.json").is_file() {
        eprintln!(
            "skipping {bench_name}: QWEN_TTS_MODEL_DIR must point to a model dir containing config.json: {}",
            path.display()
        );
        return None;
    }
    Some(path)
}

pub fn engine_config() -> EngineConfig {
    default_engine_config(8, false)
}

pub fn session_config(max_new_tokens: usize) -> SessionConfig {
    default_session_config(max_new_tokens, false)
}

pub fn synthetic_pcm(sample_count: usize) -> Vec<i16> {
    (0..sample_count)
        .map(|idx| {
            let phase = (idx % 512) as i32;
            let centered = phase - 256;
            (centered * 120) as i16
        })
        .collect()
}

pub fn synthetic_logits<B: Backend>(
    device: &B::Device,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
) -> Tensor<B, 3> {
    let data = (0..batch_size * seq_len * vocab_size)
        .map(|idx| ((idx % 257) as f32 * 0.03125) - 4.0)
        .collect::<Vec<_>>();
    Tensor::<B, 3>::from_data(
        TensorData::new(data, [batch_size, seq_len, vocab_size]),
        device,
    )
}
