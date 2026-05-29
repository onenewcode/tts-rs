use std::path::Path;

use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorData};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};
use serde::Deserialize;

use crate::{Qwen3TtsInferenceError, Qwen3TtsLoadError};

#[derive(Debug, Clone, Deserialize)]
struct SpeakerEncoderConfigManifest {
    #[serde(default = "default_mel_dim")]
    mel_dim: usize,
    #[serde(default = "default_enc_dim")]
    enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    enc_se_channels: usize,
    #[serde(default = "default_speaker_sample_rate")]
    sample_rate: u32,
}

fn default_mel_dim() -> usize {
    128
}

fn default_enc_dim() -> usize {
    1024
}

fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}

fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}

fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}

fn default_enc_attention_channels() -> usize {
    128
}

fn default_enc_res2net_scale() -> usize {
    8
}

fn default_enc_se_channels() -> usize {
    128
}

fn default_speaker_sample_rate() -> u32 {
    24_000
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfigWithSpeaker {
    #[serde(default)]
    speaker_encoder_config: Option<SpeakerEncoderConfigManifest>,
}

#[derive(Debug)]
pub(crate) struct LoadedQwen3TtsSpeakerEncoder<B: Backend> {
    encoder: SpeakerEncoderNetwork<B>,
    mel_extractor: MelSpectrogram,
    sample_rate: u32,
    device: B::Device,
}

impl<B> LoadedQwen3TtsSpeakerEncoder<B>
where
    B: Backend,
    B::Device: Clone,
{
    pub(crate) fn load(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Option<Self>, Qwen3TtsLoadError> {
        let config_path = config_path.as_ref().to_path_buf();
        let weights_path = weights_path.as_ref().to_path_buf();
        let raw = std::fs::read_to_string(&config_path).map_err(|source| {
            Qwen3TtsLoadError::CompilerConfigIo {
                path: config_path.clone(),
                source,
            }
        })?;
        let config: ModelConfigWithSpeaker = serde_json::from_str(&raw).map_err(|source| {
            Qwen3TtsLoadError::CompilerConfigParse {
                path: config_path.clone(),
                source,
            }
        })?;
        let Some(speaker_config) = config.speaker_encoder_config else {
            return Ok(None);
        };

        let mut encoder = SpeakerEncoderNetwork::new(&speaker_config, device);
        let remapper = KeyRemapper::from_patterns(vec![(r"^speaker_encoder\.(.*)$", "${1}")])
            .expect("static speaker encoder remapping must compile");
        let mut store = SafetensorsStore::from_file(&weights_path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .remap(remapper)
            .skip_enum_variants(true);
        let apply_result =
            encoder
                .load_from(&mut store)
                .map_err(|source| Qwen3TtsLoadError::Store {
                    path: weights_path.clone(),
                    source,
                })?;
        if apply_result.applied.is_empty() {
            return Ok(None);
        }

        tracing::info!(
            applied = apply_result.applied.len(),
            missing = apply_result.missing.len(),
            unused = apply_result.unused.len(),
            "loaded qwen3 tts speaker encoder weights"
        );

        Ok(Some(Self {
            encoder,
            mel_extractor: MelSpectrogram::new(MelSpectrogram::speaker_encoder()),
            sample_rate: speaker_config.sample_rate,
            device: device.clone(),
        }))
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn encode(&self, samples: &[f32]) -> Result<Vec<f32>, Qwen3TtsInferenceError> {
        let mel = self
            .mel_extractor
            .compute_for_speaker_encoder::<B>(samples, &self.device);
        let embed = self
            .encoder
            .forward(mel.unsqueeze_dim::<3>(0).cast(self.encoder.dtype()));
        embed
            .reshape([self.encoder.enc_dim])
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|source| Qwen3TtsInferenceError::TensorRead {
                message: format!("failed to read speaker embedding: {source}"),
            })
    }
}

#[derive(Debug, Clone)]
struct MelSpectrogram {
    config: MelConfig,
    mel_basis: Vec<Vec<f32>>,
    window: Vec<f32>,
}

#[derive(Debug, Clone)]
struct MelConfig {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
}

impl MelSpectrogram {
    fn speaker_encoder() -> MelConfig {
        MelConfig {
            sample_rate: 24_000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12_000.0,
        }
    }

    fn new(config: MelConfig) -> Self {
        let mel_basis = create_mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            config.fmin,
            config.fmax,
        );
        let window = hann_window(config.win_length);
        Self {
            config,
            mel_basis,
            window,
        }
    }

    fn compute_for_speaker_encoder<B: Backend>(
        &self,
        samples: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let stft = self.stft(samples);
        let mag_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im + 1e-9).sqrt())
                    .collect()
            })
            .collect();
        let mel = apply_mel_filterbank(&self.mel_basis, &mag_spec);
        let log_mel: Vec<f32> = mel
            .into_iter()
            .flat_map(|frame| frame.into_iter().map(|value| value.max(1e-5).ln()))
            .collect();
        let n_frames = log_mel.len() / self.config.n_mels;
        Tensor::<B, 2>::from_data(
            TensorData::new(log_mel, [n_frames, self.config.n_mels]),
            device,
        )
        .swap_dims(0, 1)
    }

    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let pad_length = (n_fft - hop_length) / 2;
        let mut padded = Vec::with_capacity(samples.len() + pad_length * 2);

        for index in (1..=pad_length).rev() {
            padded.push(samples[index.min(samples.len().saturating_sub(1))]);
        }
        padded.extend_from_slice(samples);
        for index in 0..pad_length {
            let sample_index = samples
                .len()
                .saturating_sub(2 + index)
                .min(samples.len().saturating_sub(1));
            padded.push(samples[sample_index]);
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        let n_frames = (padded.len() - n_fft) / hop_length + 1;
        let mut result = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let mut buffer: Vec<FftComplex<f32>> = (0..n_fft)
                .map(|offset| {
                    let sample = if offset < self.window.len() && start + offset < padded.len() {
                        padded[start + offset] * self.window[offset]
                    } else {
                        0.0
                    };
                    FftComplex::new(sample, 0.0)
                })
                .collect();
            fft.process(&mut buffer);
            result.push(
                buffer
                    .iter()
                    .take(n_fft / 2 + 1)
                    .map(|value| Complex::new(value.re, value.im))
                    .collect(),
            );
        }

        result
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            let phase = 2.0 * std::f32::consts::PI * index as f32 / size as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.068_751_74;

    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.068_751_74;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().copied().map(mel_to_hz).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|index| index as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filters = vec![vec![0.0; n_freqs]; n_mels];
    for mel_idx in 0..n_mels {
        let left = hz_points[mel_idx];
        let center = hz_points[mel_idx + 1];
        let right = hz_points[mel_idx + 2];

        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            if freq >= left && freq <= center && center > left {
                filters[mel_idx][freq_idx] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                filters[mel_idx][freq_idx] = (right - freq) / (right - center);
            }
        }

        let band_width = hz_points[mel_idx + 2] - hz_points[mel_idx];
        if band_width > 0.0 {
            let enorm = 2.0 / band_width;
            for value in &mut filters[mel_idx] {
                *value *= enorm;
            }
        }
    }
    filters
}

fn apply_mel_filterbank(mel_basis: &[Vec<f32>], power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
    power_spec
        .iter()
        .map(|frame| {
            mel_basis
                .iter()
                .map(|filter| filter.iter().zip(frame.iter()).map(|(f, p)| f * p).sum())
                .collect()
        })
        .collect()
}

#[derive(Module, Debug)]
struct SpeakerEncoderNetwork<B: Backend> {
    blocks: Vec<SpeakerEncoderBlock<B>>,
    mfa: TimeDelayNetBlock<B>,
    asp: AttentiveStatisticsPooling<B>,
    fc: Conv1d<B>,
    #[module(skip)]
    enc_dim: usize,
}

impl<B: Backend> SpeakerEncoderNetwork<B> {
    fn new(config: &SpeakerEncoderConfigManifest, device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(4);
        blocks.push(SpeakerEncoderBlock::Initial(TimeDelayNetBlock::new(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            device,
        )));
        for idx in 1..4 {
            blocks.push(SpeakerEncoderBlock::Se(SqueezeExcitationRes2NetBlock::new(
                config.enc_channels[idx],
                config.enc_kernel_sizes[idx],
                config.enc_dilations[idx],
                config.enc_res2net_scale,
                config.enc_se_channels,
                device,
            )));
        }

        let mfa_in_channels: usize = config.enc_channels[1..4].iter().sum();
        Self {
            blocks,
            mfa: TimeDelayNetBlock::new(
                mfa_in_channels,
                config.enc_channels[4],
                config.enc_kernel_sizes[4],
                config.enc_dilations[4],
                device,
            ),
            asp: AttentiveStatisticsPooling::new(
                config.enc_channels[4],
                config.enc_attention_channels,
                device,
            ),
            fc: Conv1dConfig::new(config.enc_channels[4] * 2, config.enc_dim, 1)
                .with_bias(true)
                .init(device),
            enc_dim: config.enc_dim,
        }
    }

    fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 2> {
        let SpeakerEncoderBlock::Initial(initial_tdnn) = &self.blocks[0] else {
            unreachable!("speaker encoder block 0 is always the initial TDNN")
        };
        let mut hidden = initial_tdnn.forward(mel);
        let mut outputs = Vec::with_capacity(3);
        for block in &self.blocks[1..] {
            let SpeakerEncoderBlock::Se(block) = block else {
                unreachable!("speaker encoder blocks 1..3 are SE-Res2Net blocks")
            };
            hidden = block.forward(hidden);
            outputs.push(hidden.clone());
        }
        let hidden = self.mfa.forward(Tensor::cat(outputs, 1));
        let pooled = self.asp.forward(hidden);
        self.fc.forward(pooled).squeeze_dim(2)
    }

    fn dtype(&self) -> DType {
        let SpeakerEncoderBlock::Initial(initial_tdnn) = &self.blocks[0] else {
            unreachable!("speaker encoder block 0 is always the initial TDNN")
        };
        initial_tdnn.conv.weight.val().dtype()
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum SpeakerEncoderBlock<B: Backend> {
    Initial(TimeDelayNetBlock<B>),
    Se(SqueezeExcitationRes2NetBlock<B>),
}

#[derive(Module, Debug)]
struct TimeDelayNetBlock<B: Backend> {
    conv: Conv1d<B>,
    #[module(skip)]
    pad_left: usize,
    #[module(skip)]
    pad_right: usize,
}

impl<B: Backend> TimeDelayNetBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let total_pad = dilation * (kernel_size - 1);
        Self {
            conv: Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(true)
                .init(device),
            pad_left: total_pad / 2,
            pad_right: total_pad - total_pad / 2,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(
            self.conv
                .forward(reflect_pad_1d(x, self.pad_left, self.pad_right)),
        )
    }
}

#[derive(Module, Debug)]
struct Res2NetBlock<B: Backend> {
    blocks: Vec<TimeDelayNetBlock<B>>,
    #[module(skip)]
    scale: usize,
    #[module(skip)]
    chunk_size: usize,
}

impl<B: Backend> Res2NetBlock<B> {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_size = channels / scale;
        let blocks = (0..scale - 1)
            .map(|_| TimeDelayNetBlock::new(chunk_size, chunk_size, kernel_size, dilation, device))
            .collect();
        Self {
            blocks,
            scale,
            chunk_size,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut outputs = Vec::with_capacity(self.scale);
        outputs.push(
            x.clone()
                .slice([0..x.dims()[0], 0..self.chunk_size, 0..x.dims()[2]]),
        );
        for (idx, block) in self.blocks.iter().enumerate() {
            let chunk = x.clone().slice([
                0..x.dims()[0],
                (idx + 1) * self.chunk_size..(idx + 2) * self.chunk_size,
                0..x.dims()[2],
            ]);
            let input = if idx == 0 {
                chunk
            } else {
                chunk + outputs.last().expect("previous Res2Net chunk").clone()
            };
            outputs.push(block.forward(input));
        }
        Tensor::cat(outputs, 1)
    }
}

#[derive(Module, Debug)]
struct SqueezeExcitationBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> SqueezeExcitationBlock<B> {
    fn new(channels: usize, se_channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(channels, se_channels, 1)
                .with_bias(true)
                .init(device),
            conv2: Conv1dConfig::new(se_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = x.clone().mean_dim(2);
        let scale = relu(self.conv1.forward(scale));
        let scale = sigmoid(self.conv2.forward(scale));
        x * scale
    }
}

#[derive(Module, Debug)]
struct SqueezeExcitationRes2NetBlock<B: Backend> {
    tdnn1: TimeDelayNetBlock<B>,
    res2net_block: Res2NetBlock<B>,
    tdnn2: TimeDelayNetBlock<B>,
    se_block: SqueezeExcitationBlock<B>,
}

impl<B: Backend> SqueezeExcitationRes2NetBlock<B> {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        se_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            tdnn1: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            res2net_block: Res2NetBlock::new(channels, kernel_size, dilation, scale, device),
            tdnn2: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            se_block: SqueezeExcitationBlock::new(channels, se_channels, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let hidden = self.tdnn1.forward(x);
        let hidden = self.res2net_block.forward(hidden);
        let hidden = self.tdnn2.forward(hidden);
        let hidden = self.se_block.forward(hidden);
        hidden + residual
    }
}

#[derive(Module, Debug)]
struct AttentiveStatisticsPooling<B: Backend> {
    tdnn: TimeDelayNetBlock<B>,
    conv: Conv1d<B>,
}

impl<B: Backend> AttentiveStatisticsPooling<B> {
    fn new(channels: usize, attention_channels: usize, device: &B::Device) -> Self {
        Self {
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, device),
            conv: Conv1dConfig::new(attention_channels, channels, 1)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let mean = x.clone().mean_dim(2);
        let diff = x.clone() - mean.clone();
        let var = diff.powi_scalar(2).mean_dim(2);
        let std = (var + 1e-5).sqrt();
        let attn_in = Tensor::cat(
            vec![
                x.clone(),
                mean.clone().expand([batch, channels, time]),
                std.expand([batch, channels, time]),
            ],
            1,
        );
        let attn = self.tdnn.forward(attn_in).tanh();
        let attn = softmax(self.conv.forward(attn), 2);
        let weighted_mean = (x.clone() * attn.clone()).sum_dim(2);
        let weighted_diff = x - weighted_mean.clone();
        let weighted_var = (weighted_diff.powi_scalar(2) * attn).sum_dim(2);
        let weighted_std = (weighted_var + 1e-5).sqrt();
        Tensor::cat(vec![weighted_mean, weighted_std], 1)
    }
}

fn reflect_pad_1d<B: Backend>(x: Tensor<B, 3>, pad_left: usize, pad_right: usize) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return x;
    }

    let [batch, channels, time] = x.dims();
    let mut segments = Vec::with_capacity(3);
    if pad_left > 0 {
        segments.push(
            x.clone()
                .slice([0..batch, 0..channels, 1..pad_left + 1])
                .flip([2]),
        );
    }
    segments.push(x.clone());
    if pad_right > 0 {
        segments.push(
            x.slice([0..batch, 0..channels, time - 1 - pad_right..time - 1])
                .flip([2]),
        );
    }
    Tensor::cat(segments, 2)
}
