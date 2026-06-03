mod request;

use std::any::type_name;
use std::fmt;

use burn::tensor::FloatDType;
use burn::tensor::quantization::QuantValue;
use tts_error::DiagnosticError;
use tts_infer::DriverDescriptor;
use tts_infer::DriverRegistry;
use tts_infer::PcmAudio;
use tts_infer::driver::DriverFactory;

pub use crate::loading::package::{
    Qwen3TtsArtifactsManifest, Qwen3TtsGenerationConfigManifest, Qwen3TtsGenerationConfigSource,
    Qwen3TtsPackage, Qwen3TtsPackageManifest, Qwen3TtsPackageSource,
};
pub use request::{
    BaseRequest, BaseVoiceCloneConditioning, BaseVoiceCloneReferenceAudio, CustomVoiceRequest,
    LanguageSelection, Qwen3TtsVoiceClonePrompt, Qwen3TtsVoiceClonePromptMode, QwenRequest,
};

use crate::execution::Qwen3LoadedModelInstance;

pub const DRIVER_ID: &str = "qwen3_tts";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsEngineConfig {
    pub package: Qwen3TtsPackageSource,
    pub profiling: crate::Qwen3TtsProfilingConfig,
    pub dtype: Option<Qwen3TtsModelDType>,
}

pub type Qwen3TtsLoadOptions = Qwen3TtsEngineConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Qwen3TtsModelDType {
    F64,
    F32,
    Flex32,
    F16,
    #[default]
    BF16,
    Q8F,
    Q8S,
    Q4F,
    Q4S,
    Q2F,
    Q2S,
}

impl Qwen3TtsModelDType {
    pub(crate) fn resolve(dtype: Option<Self>) -> Self {
        dtype.unwrap_or_default()
    }

    pub(crate) fn float_dtype(self) -> Option<FloatDType> {
        match self {
            Self::F64 => Some(FloatDType::F64),
            Self::F32 => Some(FloatDType::F32),
            Self::Flex32 => Some(FloatDType::Flex32),
            Self::F16 => Some(FloatDType::F16),
            Self::BF16 => Some(FloatDType::BF16),
            Self::Q8F | Self::Q8S | Self::Q4F | Self::Q4S | Self::Q2F | Self::Q2S => None,
        }
    }

    pub(crate) fn quant_value(self) -> Option<QuantValue> {
        Some(match self {
            Self::Q8F => QuantValue::Q8F,
            Self::Q8S => QuantValue::Q8S,
            Self::Q4F => QuantValue::Q4F,
            Self::Q4S => QuantValue::Q4S,
            Self::Q2F => QuantValue::Q2F,
            Self::Q2S => QuantValue::Q2S,
            Self::F64 | Self::F32 | Self::Flex32 | Self::F16 | Self::BF16 => return None,
        })
    }
}

impl fmt::Display for Qwen3TtsModelDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::Flex32 => "flex32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q8F => "q8f",
            Self::Q8S => "q8s",
            Self::Q4F => "q4f",
            Self::Q4S => "q4s",
            Self::Q2F => "q2f",
            Self::Q2S => "q2s",
        };
        f.write_str(name)
    }
}

impl std::str::FromStr for Qwen3TtsModelDType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f64" => Ok(Self::F64),
            "f32" => Ok(Self::F32),
            "flex32" => Ok(Self::Flex32),
            "f16" | "float16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "q8f" => Ok(Self::Q8F),
            "q8s" | "int8" | "qint8" => Ok(Self::Q8S),
            "q4f" => Ok(Self::Q4F),
            "q4s" | "int4" | "qint4" => Ok(Self::Q4S),
            "q2f" => Ok(Self::Q2F),
            "q2s" | "int2" | "qint2" => Ok(Self::Q2S),
            other => Err(format!(
                "unsupported dtype `{other}`; expected one of f64, f32, flex32, f16, bf16, q8f, q8s, q4f, q4s, q2f, q2s"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Qwen3TtsRunOptions {
    pub max_new_tokens: Option<usize>,
    pub sampling: Option<SamplingOverride>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingOverride {
    Literal(crate::SamplingConfig),
    GreedyFromModelDefaults,
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsEngine {
    instance: Qwen3LoadedModelInstance,
}

impl Qwen3TtsEngine {
    pub fn load(config: Qwen3TtsEngineConfig) -> Result<Self, crate::Qwen3TtsLoadError> {
        Ok(Self {
            instance: crate::execution::load_for_engine(&config)?,
        })
    }

    pub fn synthesize(
        &self,
        request: QwenRequest,
        options: Qwen3TtsRunOptions,
    ) -> Result<PcmAudio, crate::Qwen3TtsError> {
        self.instance.synthesize_audio(request, options)
    }

    pub fn create_voice_clone_prompt(
        &self,
        reference: BaseVoiceCloneReferenceAudio,
    ) -> Result<Qwen3TtsVoiceClonePrompt, crate::Qwen3TtsError> {
        self.instance.create_voice_clone_prompt(reference)
    }

    pub fn synthesize_batch<I>(
        &self,
        requests: I,
        options: Qwen3TtsRunOptions,
    ) -> Result<Vec<PcmAudio>, crate::Qwen3TtsError>
    where
        I: IntoIterator<Item = QwenRequest>,
    {
        synthesize_batch_with(requests, |request| {
            self.synthesize(request, options.clone())
        })
    }

    pub fn package(&self) -> &Qwen3TtsPackage {
        &self.instance.package
    }

    pub fn profiling(&self) -> &crate::Qwen3TtsProfilingConfig {
        &self.instance.profiling
    }
}

fn synthesize_batch_with<I, F, T, E>(requests: I, mut synthesize: F) -> Result<Vec<T>, E>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
{
    let mut outputs = Vec::new();
    for request in requests {
        outputs.push(synthesize(request)?);
    }
    Ok(outputs)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3TtsDriver;

impl DriverFactory for Qwen3TtsDriver {
    type Config = Qwen3TtsEngineConfig;

    fn descriptor(&self) -> DriverDescriptor {
        DriverDescriptor::new(
            DRIVER_ID,
            "Qwen3-TTS",
            "Qwen3-TTS local driver",
            type_name::<Qwen3TtsEngineConfig>(),
        )
    }

    fn load(
        &self,
        config: Self::Config,
    ) -> Result<Box<dyn tts_infer::driver::ErasedLoadedModel>, DiagnosticError> {
        crate::loading::load_instance(&config)
            .map(|instance| Box::new(instance) as Box<dyn tts_infer::driver::ErasedLoadedModel>)
            .map_err(DiagnosticError::from)
    }
}

pub fn register_driver(registry: &mut DriverRegistry) -> Result<(), DiagnosticError> {
    registry.register(Qwen3TtsDriver)
}

#[cfg(test)]
mod tests {
    use super::synthesize_batch_with;
    use tts_infer::PcmAudio;

    #[test]
    fn batch_wrapper_preserves_order() {
        let outputs = synthesize_batch_with([1, 2, 3], |value| {
            Ok::<_, &'static str>(PcmAudio {
                pcm_i16: vec![value],
                sample_rate: 24_000,
                channels: 1,
            })
        })
        .unwrap();

        assert_eq!(outputs[0].pcm_i16, vec![1]);
        assert_eq!(outputs[1].pcm_i16, vec![2]);
        assert_eq!(outputs[2].pcm_i16, vec![3]);
    }

    #[test]
    fn batch_wrapper_stops_on_first_error() {
        let mut seen = Vec::new();
        let error = synthesize_batch_with([1, 2, 3], |value| {
            seen.push(value);
            if value == 2 { Err("boom") } else { Ok(value) }
        })
        .unwrap_err();

        assert_eq!(error, "boom");
        assert_eq!(seen, vec![1, 2]);
    }
}
