use std::collections::HashMap;
use std::sync::Arc;

use crate::TtsModelExecutor;

#[derive(Default)]
pub struct ModelRegistry {
    executors: HashMap<String, Arc<dyn TtsModelExecutor>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        model_id: impl Into<String>,
        executor: Arc<dyn TtsModelExecutor>,
    ) -> Option<Arc<dyn TtsModelExecutor>> {
        self.executors.insert(model_id.into(), executor)
    }

    pub fn get(&self, model_id: &str) -> Option<&Arc<dyn TtsModelExecutor>> {
        self.executors.get(model_id)
    }
}
