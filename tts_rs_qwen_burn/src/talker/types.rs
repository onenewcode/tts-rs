//! Input/output types for the talker inference pipeline.
//!
//! Separated from inference logic to keep orchestration files focused.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::shared::runtime::sampling::{SamplingConfig, StoppingRules};

pub type TalkerActivations<B> = BTreeMap<String, Tensor<B, 3>>;

#[derive(Debug)]
pub struct TalkerForwardInput<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct TalkerForwardOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerDecodeInput<B: Backend> {
    pub inputs_embeds: Tensor<B, 3>,
    pub position_ids: Tensor<B, 3, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct TalkerDecodeOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerGenerateInput<B: Backend> {
    pub prefill_inputs_embeds: Tensor<B, 3>,
    pub prefill_position_ids: Tensor<B, 3, Int>,
    pub prefill_attention_mask: Option<Tensor<B, 2, Int>>,
    pub sampling: SamplingConfig,
    pub stopping: StoppingRules,
    pub suppress_token_ids: Vec<usize>,
    pub collect_step_diagnostics: bool,
}

#[derive(Debug)]
pub struct TalkerGenerateStepDiagnostic<B: Backend> {
    pub cache_len_before: usize,
    pub cache_len_after: usize,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct TalkerGenerateOutput<B: Backend> {
    pub generated_token_ids: Tensor<B, 2, Int>,
    pub step_hidden_states: Vec<Tensor<B, 2>>,
    pub prefill_logits: Tensor<B, 3>,
    pub step_logits: Vec<Tensor<B, 3>>,
    pub step_diagnostics: Vec<TalkerGenerateStepDiagnostic<B>>,
}

#[derive(Debug)]
pub struct CodePredictorTeacherForcedInput<B: Backend> {
    pub talker_hidden_states: Tensor<B, 2>,
    pub codec_ids: Tensor<B, 2, Int>,
    pub attention_mask: Option<Tensor<B, 2, Int>>,
    pub collect_activations: bool,
}

#[derive(Debug)]
pub struct CodePredictorTeacherForcedOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub activations: TalkerActivations<B>,
}

#[derive(Debug)]
pub struct CodePredictorGenerateInput<B: Backend> {
    pub talker_hidden_state: Tensor<B, 2>,
    pub base_codec_token_id: Tensor<B, 2, Int>,
    pub sampling: SamplingConfig,
    pub collect_step_diagnostics: bool,
}

#[derive(Debug)]
pub struct CodePredictorGenerateStepDiagnostic {
    pub cache_len_before: usize,
    pub cache_len_after: usize,
}

#[derive(Debug)]
pub struct CodePredictorGenerateOutput<B: Backend> {
    pub codec_ids: Tensor<B, 2, Int>,
    pub predictor_token_ids: Tensor<B, 2, Int>,
    pub step_logits: Vec<Tensor<B, 3>>,
    pub step_diagnostics: Vec<CodePredictorGenerateStepDiagnostic>,
}
