use std::io::Write;
use std::path::Path;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub fn save_wav<B: Backend>(
    waveform: &Tensor<B, 3>,
    path: impl AsRef<Path>,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_wav(waveform, &mut writer, sample_rate)
}

pub fn write_wav<B: Backend, W: Write>(
    waveform: &Tensor<B, 3>,
    writer: &mut W,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples: Vec<f32> = waveform
        .clone()
        .flatten::<1>(0, 2)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| format!("failed to read waveform: {e}"))?;

    let data_size = (samples.len() * 2) as u32;
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_size).to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&(sample_rate * 2).to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    for &sample in &samples {
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
