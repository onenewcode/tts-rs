use std::io::Write;
use std::path::Path;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::Qwen3TtsInferenceError;

pub fn save_wav<B: Backend>(
    waveform: &Tensor<B, 3>,
    path: impl AsRef<Path>,
    sample_rate: u32,
) -> Result<(), Qwen3TtsInferenceError> {
    let samples: Vec<f32> = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| Qwen3TtsInferenceError::TensorRead {
            message: format!("failed to read waveform: {e}"),
        })?;
    let pcm = samples
        .into_iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect::<Vec<_>>();
    save_pcm_wav(&pcm, path, sample_rate)
}

pub fn save_pcm_wav(
    pcm: &[i16],
    path: impl AsRef<Path>,
    sample_rate: u32,
) -> Result<(), Qwen3TtsInferenceError> {
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|source| Qwen3TtsInferenceError::Io {
        context: format!("failed to create {}", path.display()),
        source,
    })?;
    let mut writer = std::io::BufWriter::new(file);
    write_pcm_wav(pcm, &mut writer, sample_rate)
}

pub fn write_wav<B: Backend, W: Write>(
    waveform: &Tensor<B, 3>,
    writer: &mut W,
    sample_rate: u32,
) -> Result<(), Qwen3TtsInferenceError> {
    let samples: Vec<f32> = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| Qwen3TtsInferenceError::TensorRead {
            message: format!("failed to read waveform: {e}"),
        })?;
    let pcm = samples
        .into_iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect::<Vec<_>>();
    write_pcm_wav(&pcm, writer, sample_rate)
}

pub fn write_pcm_wav<W: Write>(
    pcm: &[i16],
    writer: &mut W,
    sample_rate: u32,
) -> Result<(), Qwen3TtsInferenceError> {
    let data_size = (pcm.len() * 2) as u32;
    write_all(writer, b"RIFF", "failed to write wav RIFF header")?;
    write_all(
        writer,
        &(36 + data_size).to_le_bytes(),
        "failed to write wav chunk size",
    )?;
    write_all(writer, b"WAVE", "failed to write wav format")?;
    write_all(writer, b"fmt ", "failed to write wav fmt header")?;
    write_all(writer, &16u32.to_le_bytes(), "failed to write wav fmt size")?;
    write_all(writer, &1u16.to_le_bytes(), "failed to write wav encoding")?;
    write_all(
        writer,
        &1u16.to_le_bytes(),
        "failed to write wav channel count",
    )?;
    write_all(
        writer,
        &sample_rate.to_le_bytes(),
        "failed to write wav sample rate",
    )?;
    write_all(
        writer,
        &(sample_rate * 2).to_le_bytes(),
        "failed to write wav byte rate",
    )?;
    write_all(
        writer,
        &2u16.to_le_bytes(),
        "failed to write wav block align",
    )?;
    write_all(
        writer,
        &16u16.to_le_bytes(),
        "failed to write wav bit depth",
    )?;
    write_all(writer, b"data", "failed to write wav data header")?;
    write_all(
        writer,
        &data_size.to_le_bytes(),
        "failed to write wav data size",
    )?;
    for &sample in pcm {
        write_all(
            writer,
            &sample.to_le_bytes(),
            "failed to write wav sample data",
        )?;
    }
    writer
        .flush()
        .map_err(|source| Qwen3TtsInferenceError::Io {
            context: "failed to flush wav output".to_string(),
            source,
        })?;
    Ok(())
}

fn write_all<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    context: &str,
) -> Result<(), Qwen3TtsInferenceError> {
    writer
        .write_all(bytes)
        .map_err(|source| Qwen3TtsInferenceError::Io {
            context: context.to_string(),
            source,
        })
}
