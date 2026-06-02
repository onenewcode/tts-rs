use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::{
    BaseVoiceCloneConditioning, CustomVoiceRequest, Qwen3TtsInferenceError,
    Qwen3TtsVoiceClonePrompt, Qwen3TtsVoiceClonePromptMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3TtsPromptRecipe {
    BasePlain,
    BaseVoiceCloneIcl,
    BaseVoiceCloneXVectorOnly,
    CustomVoicePlain,
    CustomVoiceInstructed,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SemanticRequestCondition {
    pub(crate) text_token_ids: Vec<i64>,
    pub(crate) instruct_token_ids: Option<Vec<i64>>,
    pub(crate) voice_clone: Option<CompiledVoiceCloneCondition>,
    pub(crate) controls: ProfileControlIds,
    pub(crate) max_new_tokens: usize,
    pub(crate) codec_eos_token_id: usize,
    pub(crate) sampling: crate::SamplingConfig,
    pub(crate) prompt_recipe: Qwen3TtsPromptRecipe,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledVoiceCloneCondition {
    pub(crate) source: CompiledVoiceCloneConditionSource,
    pub(crate) ref_text_token_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CompiledVoiceCloneConditionSource {
    Prompt(Qwen3TtsVoiceClonePrompt),
    ReferenceAudio(CompiledReferenceAudioVoiceClone),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledReferenceAudioVoiceClone {
    pub(crate) path: PathBuf,
    pub(crate) transcript: Option<String>,
    pub(crate) x_vector_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileControlIds {
    pub(crate) tts_bos_token_id: i64,
    pub(crate) tts_eos_token_id: i64,
    pub(crate) tts_pad_token_id: i64,
    pub(crate) codec_bos_id: i64,
    pub(crate) codec_pad_id: i64,
    pub(crate) codec_prefix_ids: Vec<i64>,
}

pub(crate) struct CompileProfileConditionInput<'a> {
    pub(crate) prompt: &'a str,
    pub(crate) instruct_prompt: Option<&'a str>,
    pub(crate) voice_clone: Option<CompiledVoiceCloneCondition>,
    pub(crate) ref_prompt: Option<&'a str>,
    pub(crate) prompt_recipe: Qwen3TtsPromptRecipe,
    pub(crate) controls: ProfileControlIds,
    pub(crate) max_new_tokens: usize,
    pub(crate) codec_eos_token_id: usize,
    pub(crate) sampling: crate::SamplingConfig,
}

fn build_assistant_prompt(text: &str) -> String {
    format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        text
    )
}

fn build_ref_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{}<|im_end|>\n", text)
}

fn build_instruct_prompt(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text)
}

pub(crate) fn resolve_base_prompt_recipe(
    request: &crate::BaseRequest,
) -> Result<
    (
        Qwen3TtsPromptRecipe,
        String,
        Option<String>,
        Option<CompiledVoiceCloneCondition>,
    ),
    Qwen3TtsInferenceError,
> {
    let prompt = build_assistant_prompt(&request.text);
    let Some(voice_clone) = request.voice_clone.as_ref() else {
        return Ok((Qwen3TtsPromptRecipe::BasePlain, prompt, None, None));
    };

    let (recipe, ref_prompt, compiled) = match voice_clone {
        BaseVoiceCloneConditioning::ReferenceAudio(reference) => {
            let transcript = reference
                .transcript
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let recipe = if reference.x_vector_only {
                Qwen3TtsPromptRecipe::BaseVoiceCloneXVectorOnly
            } else {
                Qwen3TtsPromptRecipe::BaseVoiceCloneIcl
            };
            if matches!(recipe, Qwen3TtsPromptRecipe::BaseVoiceCloneIcl) && transcript.is_none() {
                return Err(Qwen3TtsInferenceError::InvalidInput {
                    message: "ref_text is required when x_vector_only is false".to_string(),
                });
            }
            (
                recipe,
                transcript.as_deref().map(build_ref_prompt),
                CompiledVoiceCloneCondition {
                    source: CompiledVoiceCloneConditionSource::ReferenceAudio(
                        CompiledReferenceAudioVoiceClone {
                            path: reference.path.clone(),
                            transcript,
                            x_vector_only: reference.x_vector_only,
                        },
                    ),
                    ref_text_token_ids: None,
                },
            )
        }
        BaseVoiceCloneConditioning::Prompt(prompt) => match prompt.mode {
            Qwen3TtsVoiceClonePromptMode::XVectorOnly => (
                Qwen3TtsPromptRecipe::BaseVoiceCloneXVectorOnly,
                None,
                CompiledVoiceCloneCondition {
                    source: CompiledVoiceCloneConditionSource::Prompt(prompt.clone()),
                    ref_text_token_ids: None,
                },
            ),
            Qwen3TtsVoiceClonePromptMode::Icl => {
                if prompt
                    .ref_codec_token_ids
                    .as_ref()
                    .is_none_or(|codes| codes.is_empty())
                {
                    return Err(Qwen3TtsInferenceError::InvalidInput {
                        message: "voice clone prompt in ICL mode requires ref codec frames"
                            .to_string(),
                    });
                }
                let transcript = prompt
                    .transcript
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                        message: "voice clone prompt in ICL mode requires ref_text".to_string(),
                    })?;
                (
                    Qwen3TtsPromptRecipe::BaseVoiceCloneIcl,
                    Some(build_ref_prompt(transcript)),
                    CompiledVoiceCloneCondition {
                        source: CompiledVoiceCloneConditionSource::Prompt(prompt.clone()),
                        ref_text_token_ids: None,
                    },
                )
            }
        },
    };

    Ok((recipe, prompt, ref_prompt, Some(compiled)))
}

pub(crate) fn resolve_custom_voice_prompt_recipe(
    request: &CustomVoiceRequest,
) -> Result<(Qwen3TtsPromptRecipe, String, Option<String>), Qwen3TtsInferenceError> {
    if request
        .speaker
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_none()
    {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: "custom-voice requests require a non-empty speaker".to_string(),
        });
    }

    match request
        .instruct
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(instruct) => Ok((
            Qwen3TtsPromptRecipe::CustomVoiceInstructed,
            build_assistant_prompt(&request.text),
            Some(build_instruct_prompt(instruct)),
        )),
        None => Ok((
            Qwen3TtsPromptRecipe::CustomVoicePlain,
            build_assistant_prompt(&request.text),
            None,
        )),
    }
}

pub(crate) fn compile_profile_condition(
    tokenizer: &Tokenizer,
    input: CompileProfileConditionInput<'_>,
) -> Result<SemanticRequestCondition, Qwen3TtsInferenceError> {
    let text_token_ids = tokenize_prompt(tokenizer, input.prompt)?;
    if text_token_ids.len() < 8 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "qwen prompt tokenization is too short: {} tokens",
                text_token_ids.len()
            ),
        });
    }

    let mut voice_clone = input.voice_clone;
    if let Some(voice_clone) = voice_clone.as_mut()
        && matches!(input.prompt_recipe, Qwen3TtsPromptRecipe::BaseVoiceCloneIcl)
    {
        let ref_prompt = input
            .ref_prompt
            .ok_or_else(|| Qwen3TtsInferenceError::InvalidInput {
                message: "voice clone ICL recipe requires a tokenizable ref prompt".to_string(),
            })?;
        voice_clone.ref_text_token_ids = Some(tokenize_prompt(tokenizer, ref_prompt)?);
    }

    Ok(SemanticRequestCondition {
        text_token_ids,
        instruct_token_ids: input
            .instruct_prompt
            .map(|prompt| tokenize_prompt(tokenizer, prompt))
            .transpose()?,
        voice_clone,
        controls: input.controls,
        max_new_tokens: input.max_new_tokens,
        codec_eos_token_id: input.codec_eos_token_id,
        sampling: input.sampling,
        prompt_recipe: input.prompt_recipe,
    })
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
) -> Result<Vec<i64>, Qwen3TtsInferenceError> {
    Ok(tokenizer
        .encode(prompt, false)
        .map_err(|source| Qwen3TtsInferenceError::Tokenizer { source })?
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>())
}
