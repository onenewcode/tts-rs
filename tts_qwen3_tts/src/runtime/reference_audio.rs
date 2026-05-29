use std::path::Path;

use crate::Qwen3TtsInferenceError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedReferenceAudio {
    pub(crate) sample_rate: u32,
    pub(crate) samples: Vec<f32>,
}

pub(crate) fn load_reference_audio(
    path: &Path,
    target_sample_rate: u32,
) -> Result<PreparedReferenceAudio, Qwen3TtsInferenceError> {
    let mut reader =
        hound::WavReader::open(path).map_err(|source| Qwen3TtsInferenceError::AudioDecode {
            message: format!("failed to open wav {}: {source}", path.display()),
        })?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("wav {} has zero channels", path.display()),
        });
    }

    let interleaved = read_wav_samples(&mut reader, &spec)?;
    if interleaved.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("wav {} contains no audio samples", path.display()),
        });
    }

    let mono = mixdown_to_mono(&interleaved, channels);
    if mono.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!("wav {} contains no complete audio frames", path.display()),
        });
    }

    let resampled = if spec.sample_rate == target_sample_rate {
        mono
    } else {
        resample_linear(&mono, spec.sample_rate, target_sample_rate)
    };
    if resampled.is_empty() {
        return Err(Qwen3TtsInferenceError::InvalidInput {
            message: format!(
                "wav {} produced no samples after resampling to {} Hz",
                path.display(),
                target_sample_rate
            ),
        });
    }

    Ok(PreparedReferenceAudio {
        sample_rate: target_sample_rate,
        samples: resampled,
    })
}

fn read_wav_samples(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: &hound::WavSpec,
) -> Result<Vec<f32>, Qwen3TtsInferenceError> {
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map_err(|source| Qwen3TtsInferenceError::AudioDecode {
                    message: format!("failed to decode float32 wav payload: {source}"),
                })
            })
            .collect(),
        (hound::SampleFormat::Int, 16) => read_pcm_samples(reader, 32768.0),
        (hound::SampleFormat::Int, 24) => read_pcm_samples(reader, 8_388_608.0),
        (hound::SampleFormat::Int, 32) => read_pcm_samples(reader, 2_147_483_648.0),
        (format, bits) => Err(Qwen3TtsInferenceError::AudioDecode {
            message: format!("unsupported wav format: {format:?} / {bits}-bit"),
        }),
    }
}

fn read_pcm_samples(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    scale: f32,
) -> Result<Vec<f32>, Qwen3TtsInferenceError> {
    reader
        .samples::<i32>()
        .map(|sample| {
            sample
                .map(|value| (value as f32 / scale).clamp(-1.0, 1.0))
                .map_err(|source| Qwen3TtsInferenceError::AudioDecode {
                    message: format!("failed to decode integer wav payload: {source}"),
                })
        })
        .collect()
}

fn mixdown_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

pub(crate) fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate || samples.len() == 1 {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as u64 * u64::from(target_rate)) + u64::from(source_rate / 2))
        / u64::from(source_rate);
    let output_len = output_len.max(1) as usize;
    let scale = source_rate as f64 / target_rate as f64;

    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = left.saturating_add(1).min(samples.len().saturating_sub(1));
            let mix = (position - left as f64) as f32;
            samples[left] * (1.0 - mix) + samples[right] * mix
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resample_changes_length_for_new_sample_rate() {
        let source = vec![0.0, 0.5, -0.5, 1.0];
        let resampled = resample_linear(&source, 4, 8);
        assert_eq!(resampled.len(), 8);
        assert!(resampled.iter().any(|sample| *sample > 0.0));
    }

    #[test]
    fn linear_resample_returns_empty_for_invalid_rates() {
        assert!(resample_linear(&[0.0, 1.0], 0, 24_000).is_empty());
        assert!(resample_linear(&[0.0, 1.0], 24_000, 0).is_empty());
    }

    #[test]
    fn load_reference_audio_mixdowns_stereo_and_resamples() {
        let path = temp_wav_path("stereo-resample");
        write_wav_i16(
            &path,
            48_000,
            2,
            &[500, -500, 1_000, -1_000, 1_500, -1_500, 2_000, -2_000],
        );

        let audio = load_reference_audio(&path, 24_000).unwrap();

        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), 2);
        assert!(audio.samples.iter().all(|sample| sample.abs() < 0.01));
    }

    #[test]
    fn load_reference_audio_rejects_empty_payload() {
        let path = temp_wav_path("empty");
        write_wav_i16(&path, 24_000, 1, &[]);

        let error = load_reference_audio(&path, 24_000).unwrap_err();
        assert!(error.to_string().contains("contains no audio samples"));
    }

    fn temp_wav_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tts-rs-reference-audio-{label}-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos()
        ))
    }

    fn write_wav_i16(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for sample in samples {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }
}
