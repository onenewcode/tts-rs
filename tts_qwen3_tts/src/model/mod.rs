use std::sync::Arc;

use tts_infer::{LoadedModel, ModelSession, SessionStep};

use crate::{
    Qwen3TtsBackend, Qwen3TtsInferenceError, Qwen3TtsPackage, Qwen3TtsProfilingConfig,
    Qwen3TtsRequestCompiler, Qwen3TtsRunOptions, QwenRequest,
};
use crate::releases::QwenProfile;

#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsModelInner {
    pub(crate) package: Qwen3TtsPackage,
    pub(crate) backend: Qwen3TtsBackend,
    pub(crate) profiling: Qwen3TtsProfilingConfig,
    pub(crate) compiler: Qwen3TtsRequestCompiler,
}

#[derive(Debug, Clone)]
pub(crate) enum Qwen3TtsLoadedModel {
    Flex(Arc<Qwen3TtsModelInner>),
    Wgpu(Arc<Qwen3TtsModelInner>),
    Cuda(Arc<Qwen3TtsModelInner>),
    Rocm(Arc<Qwen3TtsModelInner>),
    Metal(Arc<Qwen3TtsModelInner>),
    Vulkan(Arc<Qwen3TtsModelInner>),
    WebGpu(Arc<Qwen3TtsModelInner>),
}

impl Qwen3TtsLoadedModel {
    pub(crate) fn new(inner: Qwen3TtsModelInner) -> Self {
        let backend = inner.backend;
        let inner = Arc::new(inner);
        match backend {
            Qwen3TtsBackend::Flex => Self::Flex(inner),
            Qwen3TtsBackend::Wgpu => Self::Wgpu(inner),
            Qwen3TtsBackend::Cuda => Self::Cuda(inner),
            Qwen3TtsBackend::Rocm => Self::Rocm(inner),
            Qwen3TtsBackend::Metal => Self::Metal(inner),
            Qwen3TtsBackend::Vulkan => Self::Vulkan(inner),
            Qwen3TtsBackend::WebGpu => Self::WebGpu(inner),
        }
    }

    fn inner(&self) -> &Arc<Qwen3TtsModelInner> {
        match self {
            Self::Flex(inner)
            | Self::Wgpu(inner)
            | Self::Cuda(inner)
            | Self::Rocm(inner)
            | Self::Metal(inner)
            | Self::Vulkan(inner)
            | Self::WebGpu(inner) => inner,
        }
    }
}

impl LoadedModel for Qwen3TtsLoadedModel {
    type Request = QwenRequest;
    type RunOptions = Qwen3TtsRunOptions;
    type Session = Qwen3TtsSession;
    type Error = Qwen3TtsInferenceError;

    fn start_session(
        &self,
        request: Self::Request,
        options: Self::RunOptions,
    ) -> Result<Self::Session, Self::Error> {
        let inner = Arc::clone(self.inner());
        let condition = inner.compiler.compile_request(&request)?;
        Ok(Qwen3TtsSession::new(inner, request, options, condition))
    }
}

#[derive(Debug)]
pub(crate) enum Qwen3TtsSession {
    Flex(SessionImpl),
    Wgpu(SessionImpl),
    Cuda(SessionImpl),
    Rocm(SessionImpl),
    Metal(SessionImpl),
    Vulkan(SessionImpl),
    WebGpu(SessionImpl),
}

impl Qwen3TtsSession {
    fn new(
        inner: Arc<Qwen3TtsModelInner>,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
        condition: crate::compiler::SemanticRequestCondition,
    ) -> Self {
        let backend = inner.backend;
        let session = SessionImpl {
            inner,
            request,
            options,
            condition,
            terminal_reached: false,
            runtime: RuntimeState::Pending,
        };
        match backend {
            Qwen3TtsBackend::Flex => Self::Flex(session),
            Qwen3TtsBackend::Wgpu => Self::Wgpu(session),
            Qwen3TtsBackend::Cuda => Self::Cuda(session),
            Qwen3TtsBackend::Rocm => Self::Rocm(session),
            Qwen3TtsBackend::Metal => Self::Metal(session),
            Qwen3TtsBackend::Vulkan => Self::Vulkan(session),
            Qwen3TtsBackend::WebGpu => Self::WebGpu(session),
        }
    }

    fn step_impl(session: &mut SessionImpl) -> Result<SessionStep, Qwen3TtsInferenceError> {
        if matches!(session.runtime, RuntimeState::Pending) {
            session.runtime = RuntimeState::Active(crate::runtime::executor::start_model_run(
                model_dir(&session.inner.package),
                profile_for_request(&session.request),
                session.inner.backend,
                &session.request,
                &session.inner.profiling,
                &session.options,
            )?);
        }

        let RuntimeState::Active(run) = &mut session.runtime else {
            unreachable!("runtime should be active before stepping");
        };

        let step = run.step()?;
        if step.finished {
            session.terminal_reached = true;
            Ok(SessionStep::Finished)
        } else {
            Ok(SessionStep::Advanced)
        }
    }

    fn finish_impl(session: SessionImpl) -> Result<tts_infer::PcmAudio, Qwen3TtsInferenceError> {
        let _ = session.condition.prompt.len();
        match session.runtime {
            RuntimeState::Active(run) => run.finish(),
            RuntimeState::Pending => Err(Qwen3TtsInferenceError::InvalidInput {
                message: "runtime session did not reach a terminal state".to_string(),
            }),
        }
    }
}

impl ModelSession for Qwen3TtsSession {
    type Error = Qwen3TtsInferenceError;

    fn step(&mut self) -> Result<SessionStep, Self::Error> {
        match self {
            Self::Flex(session)
            | Self::Wgpu(session)
            | Self::Cuda(session)
            | Self::Rocm(session)
            | Self::Metal(session)
            | Self::Vulkan(session)
            | Self::WebGpu(session) => Self::step_impl(session),
        }
    }

    fn finish(self) -> Result<tts_infer::PcmAudio, Self::Error> {
        match self {
            Self::Flex(session)
            | Self::Wgpu(session)
            | Self::Cuda(session)
            | Self::Rocm(session)
            | Self::Metal(session)
            | Self::Vulkan(session)
            | Self::WebGpu(session) => Self::finish_impl(session),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SessionImpl {
    inner: Arc<Qwen3TtsModelInner>,
    request: QwenRequest,
    options: Qwen3TtsRunOptions,
    condition: crate::compiler::SemanticRequestCondition,
    terminal_reached: bool,
    runtime: RuntimeState,
}

#[derive(Debug)]
enum RuntimeState {
    Pending,
    Active(crate::runtime::executor::QwenModelRun),
}

fn profile_for_request(request: &QwenRequest) -> QwenProfile {
    match request {
        QwenRequest::Base(_) => QwenProfile::Base,
        QwenRequest::CustomVoice(_) => QwenProfile::CustomVoice,
    }
}

fn model_dir(package: &Qwen3TtsPackage) -> &std::path::Path {
    package
        .tokenizer_path
        .parent()
        .unwrap_or(package.package_root.as_path())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BaseRequest, Qwen3TtsPackageProfiles, Qwen3TtsProfilePackage, QwenRequest,
    };

    #[test]
    fn loaded_model_starts_session_with_compiled_request() {
        let model = Qwen3TtsLoadedModel::new(fixture_model_inner());

        let session = model
            .start_session(QwenRequest::Base(BaseRequest::new("hello")), Qwen3TtsRunOptions::default())
            .unwrap();

        match session {
            Qwen3TtsSession::Flex(session) => {
                assert_eq!(
                    session.condition.prompt,
                    "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
                );
                assert_eq!(session.condition.controls.codec_prefix_ids, vec![2051, 2052, 2053, 2049, 2048]);
            }
            _ => panic!("expected flex session"),
        }
    }

    fn fixture_model_inner() -> Qwen3TtsModelInner {
        let temp = std::env::temp_dir().join(format!(
            "tts-rs-qwen3-model-backbone-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos()
        ));
        std::fs::create_dir_all(temp.join("profiles/base")).unwrap();
        std::fs::create_dir_all(temp.join("profiles/custom_voice")).unwrap();
        std::fs::write(temp.join("profiles/base/generation_config.json"), GENERATION_CONFIG_JSON).unwrap();
        std::fs::write(temp.join("profiles/base/control_config.json"), CONTROL_CONFIG_JSON).unwrap();
        std::fs::write(
            temp.join("profiles/custom_voice/generation_config.json"),
            GENERATION_CONFIG_JSON,
        )
        .unwrap();
        std::fs::write(
            temp.join("profiles/custom_voice/control_config.json"),
            CONTROL_CONFIG_JSON,
        )
        .unwrap();

        let package = Qwen3TtsPackage {
            package_root: temp.clone(),
            name: "fixture".to_string(),
            tokenizer_path: temp.join("tokenizer.json"),
            talker_config_path: temp.join("configs/talker.json"),
            talker_weights_path: temp.join("weights/talker.safetensors"),
            codec_config_path: temp.join("configs/codec.json"),
            codec_weights_path: temp.join("weights/codec.safetensors"),
            profiles: Qwen3TtsPackageProfiles {
                base: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/base/generation_config.json"),
                    control_config_path: temp.join("profiles/base/control_config.json"),
                }),
                custom_voice: Some(Qwen3TtsProfilePackage {
                    generation_config_path: temp.join("profiles/custom_voice/generation_config.json"),
                    control_config_path: temp.join("profiles/custom_voice/control_config.json"),
                }),
            },
        };
        let compiler = Qwen3TtsRequestCompiler::load(&package).unwrap();

        Qwen3TtsModelInner {
            package,
            backend: Qwen3TtsBackend::Flex,
            profiling: Qwen3TtsProfilingConfig::default(),
            compiler,
        }
    }

    const GENERATION_CONFIG_JSON: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_new_tokens": 8192
}"#;

    const CONTROL_CONFIG_JSON: &str = r#"{
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "codec_bos_id": 2048,
  "codec_eos_token_id": 2150,
  "codec_pad_id": 2049,
  "codec_think_id": 2050,
  "codec_nothink_id": 2051,
  "codec_think_bos_id": 2052,
  "codec_think_eos_id": 2053,
  "codec_language_id": {"zh": 3001},
  "spk_id": {"chelsie": 4001},
  "spk_is_dialect": {"chelsie": "zh"}
}"#;
}
