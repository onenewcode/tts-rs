use std::collections::HashMap;
use std::sync::Arc;

use crate::TtsModelAdapter;

#[derive(Default)]
pub struct ModelRegistry {
    adapters: HashMap<String, Arc<dyn TtsModelAdapter>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        model_id: impl Into<String>,
        adapter: Arc<dyn TtsModelAdapter>,
    ) -> Option<Arc<dyn TtsModelAdapter>> {
        self.adapters.insert(model_id.into(), adapter)
    }

    pub fn get(&self, model_id: &str) -> Option<&Arc<dyn TtsModelAdapter>> {
        self.adapters.get(model_id)
    }
}
