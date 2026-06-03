use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};

use crate::model::speaker::config::SpeakerEncoderConfig;

#[derive(Debug, Clone)]
pub(crate) struct MelSpectrogram {
    config: MelConfig,
    mel_basis: Vec<Vec<f32>>,
    window: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct MelConfig {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
}

impl MelSpectrogram {
    pub(crate) fn from_speaker_encoder_config(config: &SpeakerEncoderConfig) -> Self {
        Self::new(MelConfig::for_speaker_encoder(config))
    }
}

impl MelConfig {
    fn for_speaker_encoder(config: &SpeakerEncoderConfig) -> Self {
        MelConfig {
            sample_rate: config.sample_rate,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: config.mel_dim,
            fmin: 0.0,
            fmax: config.sample_rate as f32 / 2.0,
        }
    }
}

impl MelSpectrogram {
    pub(crate) fn new(config: MelConfig) -> Self {
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

    pub(crate) fn compute_for_speaker_encoder<B: Backend>(
        &self,
        samples: &[f32],
        dtype: DType,
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
        Tensor::<B, 1>::from_data(log_mel.as_slice(), (device, DType::F32))
            .cast(dtype)
            .reshape([n_frames, self.config.n_mels])
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
            let mut buffer: Vec<Complex<f32>> = (0..n_fft)
                .map(|offset| {
                    let sample = if offset < self.window.len() && start + offset < padded.len() {
                        padded[start + offset] * self.window[offset]
                    } else {
                        0.0
                    };
                    Complex::new(sample, 0.0)
                })
                .collect();
            fft.process(&mut buffer);
            result.push(buffer.iter().take(n_fft / 2 + 1).copied().collect());
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
    const MEL_SCALE: f32 = 2_595.0;
    const MEL_BREAK_FREQUENCY_HZ: f32 = 700.0;
    MEL_SCALE * (1.0 + hz / MEL_BREAK_FREQUENCY_HZ).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    const MEL_SCALE: f32 = 2_595.0;
    const MEL_BREAK_FREQUENCY_HZ: f32 = 700.0;
    MEL_BREAK_FREQUENCY_HZ * (10_f32.powf(mel / MEL_SCALE) - 1.0)
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
