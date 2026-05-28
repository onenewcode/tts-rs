use std::io::Write;
use std::path::Path;

pub fn save_pcm_wav(pcm: &[i16], path: impl AsRef<Path>, sample_rate: u32) -> std::io::Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_pcm_wav(pcm, &mut writer, sample_rate)
}

pub fn write_pcm_wav<W: Write>(
    pcm: &[i16],
    writer: &mut W,
    sample_rate: u32,
) -> std::io::Result<()> {
    let data_size = (pcm.len() * 2) as u32;
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
    for &sample in pcm {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()
}
