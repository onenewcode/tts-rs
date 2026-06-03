use std::time::Duration;

use crate::PcmAudio;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesisResult {
    pub audio: PcmAudio,
    pub instance_id: u64,
    pub driver_id: String,
    pub elapsed: Duration,
}
