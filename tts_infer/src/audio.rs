use std::io::Write;
use std::path::Path;

// TODO 为什么不使用houn
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmAudio {
    pub pcm_i16: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl PcmAudio {
    pub fn save_wav(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = std::io::BufWriter::new(file);
        self.write_wav(&mut writer)
    }

    pub fn write_wav<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let bytes_per_sample = std::mem::size_of::<i16>() as u32;
        let data_size = self.pcm_i16.len() as u32 * bytes_per_sample;
        let block_align = self.channels * bytes_per_sample as u16;
        let byte_rate = self.sample_rate * u32::from(block_align);

        writer.write_all(b"RIFF")?;
        writer.write_all(&(36 + data_size).to_le_bytes())?;
        writer.write_all(b"WAVE")?;
        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?;
        writer.write_all(&1u16.to_le_bytes())?;
        writer.write_all(&self.channels.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&16u16.to_le_bytes())?;
        writer.write_all(b"data")?;
        writer.write_all(&data_size.to_le_bytes())?;
        for sample in &self.pcm_i16 {
            writer.write_all(&sample.to_le_bytes())?;
        }
        writer.flush()
    }
}
