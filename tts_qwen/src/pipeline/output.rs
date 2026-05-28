#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub pcm: Vec<i16>,
    pub sample_rate: u32,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    CodecChunk { steps: usize },
    AudioChunk(AudioChunk),
    Finished,
}

#[derive(Debug, Clone, Default)]
pub struct PendingAudio {
    pub emitted_steps: usize,
    pub emitted_samples: usize,
}

#[derive(Debug, Clone)]
pub struct FinishedSession {
    pub sample_rate: u32,
    pub waveform_pcm: Vec<i16>,
}
