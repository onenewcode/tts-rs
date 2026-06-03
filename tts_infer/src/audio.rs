use std::io::{Cursor, Seek, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmAudio {
    pub pcm_i16: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl PcmAudio {
    pub fn save_wav(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path.as_ref())?;
        self.write_wav_seek(std::io::BufWriter::new(file))
    }

    pub fn write_wav<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let mut buffer = Cursor::new(Vec::with_capacity(44 + self.pcm_i16.len() * 2));
        self.write_wav_seek(&mut buffer)?;
        writer.write_all(buffer.get_ref())?;
        writer.flush()
    }

    fn write_wav_seek<W: Write + Seek>(&self, writer: W) -> std::io::Result<()> {
        let mut wav =
            hound::WavWriter::new(writer, self.wav_spec()).map_err(|error| match error {
                hound::Error::IoError(source) => source,
                other => std::io::Error::new(std::io::ErrorKind::InvalidData, other),
            })?;
        for sample in &self.pcm_i16 {
            wav.write_sample(*sample).map_err(|error| match error {
                hound::Error::IoError(source) => source,
                other => std::io::Error::new(std::io::ErrorKind::InvalidData, other),
            })?;
        }
        wav.finalize().map_err(|error| match error {
            hound::Error::IoError(source) => source,
            other => std::io::Error::new(std::io::ErrorKind::InvalidData, other),
        })
    }

    fn wav_spec(&self) -> hound::WavSpec {
        hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }
}
