use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub supports_base_synthesis: bool,
    pub supports_custom_voice: bool,
    pub supports_voice_clone: bool,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub extensions: BTreeMap<String, String>,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            supports_base_synthesis: true,
            supports_custom_voice: false,
            supports_voice_clone: false,
            sample_rate_hz: 0,
            channels: 0,
            extensions: BTreeMap::new(),
        }
    }
}

impl ModelCapabilities {
    pub fn builder() -> ModelCapabilitiesBuilder {
        ModelCapabilitiesBuilder::default()
    }
}

#[derive(Debug, Default, Clone)]
pub struct ModelCapabilitiesBuilder {
    inner: ModelCapabilities,
}

impl ModelCapabilitiesBuilder {
    pub fn supports_base_synthesis(mut self, value: bool) -> Self {
        self.inner.supports_base_synthesis = value;
        self
    }

    pub fn supports_custom_voice(mut self, value: bool) -> Self {
        self.inner.supports_custom_voice = value;
        self
    }

    pub fn supports_voice_clone(mut self, value: bool) -> Self {
        self.inner.supports_voice_clone = value;
        self
    }

    pub fn sample_rate_hz(mut self, value: u32) -> Self {
        self.inner.sample_rate_hz = value;
        self
    }

    pub fn channels(mut self, value: u16) -> Self {
        self.inner.channels = value;
        self
    }

    pub fn extension(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner.extensions.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> ModelCapabilities {
        self.inner
    }
}
