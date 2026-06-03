mod audio;
mod capabilities;
pub mod driver;
mod manager;
mod result;

pub use audio::PcmAudio;
pub use capabilities::{ModelCapabilities, ModelCapabilitiesBuilder};
pub use manager::{DriverDescriptor, DriverRegistry, LoadedModelHandle, ModelManager, ModelState};
pub use result::SynthesisResult;
