use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use tts_error::DiagnosticError;

use crate::ModelCapabilities;
use crate::driver::{DriverFactory, ErasedLoadedModel, SharedDriverFactory};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverDescriptor {
    driver_id: String,
    display_name: String,
    summary: String,
    config_type: String,
}

impl DriverDescriptor {
    pub fn new(
        driver_id: impl Into<String>,
        display_name: impl Into<String>,
        summary: impl Into<String>,
        config_type: impl Into<String>,
    ) -> Self {
        Self {
            driver_id: driver_id.into(),
            display_name: display_name.into(),
            summary: summary.into(),
            config_type: config_type.into(),
        }
    }

    pub fn driver_id(&self) -> &str {
        &self.driver_id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }

    pub fn config_type(&self) -> &str {
        &self.config_type
    }
}

#[derive(Clone, Default)]
pub struct DriverRegistry {
    drivers: BTreeMap<String, SharedDriverFactory>,
}

impl DriverRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<D>(&mut self, driver: D) -> Result<(), DiagnosticError>
    where
        D: DriverFactory,
    {
        let descriptor = driver.descriptor();
        let driver_id = descriptor.driver_id().to_string();
        if self.drivers.contains_key(&driver_id) {
            return Err(DiagnosticError::conflict(
                "driver.duplicate_registration",
                format!("driver `{driver_id}` is already registered"),
            ));
        }

        self.drivers.insert(driver_id, Arc::new(driver));
        Ok(())
    }

    pub fn descriptors(&self) -> Vec<DriverDescriptor> {
        self.drivers
            .values()
            .map(|driver| driver.descriptor())
            .collect()
    }

    pub(crate) fn get(&self, driver_id: &str) -> Option<SharedDriverFactory> {
        self.drivers.get(driver_id).cloned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Ready,
    Busy,
    Closed,
}

#[derive(Clone)]
pub struct LoadedModelHandle {
    inner: Arc<LoadedModelHandleInner>,
}

impl std::fmt::Debug for LoadedModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModelHandle")
            .field("instance_id", &self.instance_id())
            .field("driver_id", &self.driver_id())
            .field("state", &self.state())
            .finish()
    }
}

struct LoadedModelHandleInner {
    instance_id: u64,
    driver_id: String,
    capabilities: ModelCapabilities,
    model: Box<dyn ErasedLoadedModel>,
    state: Mutex<LifecycleState>,
    state_changed: Condvar,
}

#[derive(Debug, Clone, Copy)]
struct LifecycleState {
    state: ModelState,
    accepting_new_work: bool,
}

impl LoadedModelHandle {
    pub(crate) fn new(instance_id: u64, model: Box<dyn ErasedLoadedModel>) -> Self {
        let driver_id = model.driver_id().to_string();
        let capabilities = model.capabilities();
        Self {
            inner: Arc::new(LoadedModelHandleInner {
                instance_id,
                driver_id,
                capabilities,
                model,
                state: Mutex::new(LifecycleState {
                    state: ModelState::Ready,
                    accepting_new_work: true,
                }),
                state_changed: Condvar::new(),
            }),
        }
    }

    pub fn instance_id(&self) -> u64 {
        self.inner.instance_id
    }

    pub fn driver_id(&self) -> &str {
        &self.inner.driver_id
    }

    pub fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities.clone()
    }

    pub fn state(&self) -> ModelState {
        self.inner.state.lock().unwrap().state
    }

    pub fn close(&self) -> Result<(), DiagnosticError> {
        let mut lifecycle = self.inner.state.lock().unwrap();
        if lifecycle.state == ModelState::Closed {
            return Ok(());
        }

        lifecycle.accepting_new_work = false;
        while lifecycle.state == ModelState::Busy {
            lifecycle = self.inner.state_changed.wait(lifecycle).unwrap();
        }

        lifecycle.state = ModelState::Closed;
        self.inner.state_changed.notify_all();
        Ok(())
    }

    pub fn with_model_as<T, F, R>(&self, f: F) -> Result<R, DiagnosticError>
    where
        T: Any + Send + Sync + 'static,
        F: FnOnce(&T) -> R,
    {
        let mut lifecycle = self.inner.state.lock().unwrap();
        loop {
            match lifecycle.state {
                ModelState::Closed => {
                    return Err(DiagnosticError::conflict(
                        "model.closed",
                        format!(
                            "model instance {} is already closed",
                            self.inner.instance_id
                        ),
                    ));
                }
                ModelState::Ready if lifecycle.accepting_new_work => {
                    lifecycle.state = ModelState::Busy;
                    break;
                }
                ModelState::Ready => {
                    return Err(DiagnosticError::conflict(
                        "model.closing",
                        format!(
                            "model instance {} is closing and rejects new work",
                            self.inner.instance_id
                        ),
                    ));
                }
                ModelState::Busy => {
                    lifecycle = self.inner.state_changed.wait(lifecycle).unwrap();
                }
            }
        }
        drop(lifecycle);

        let guard = ExecutionGuard { handle: self };
        let model = self
            .inner
            .model
            .as_any()
            .downcast_ref::<T>()
            .ok_or_else(|| {
                DiagnosticError::invalid_argument(
                    "model.type_mismatch",
                    format!(
                        "model instance {} is not a `{}`",
                        self.inner.instance_id,
                        std::any::type_name::<T>()
                    ),
                )
            })?;
        let result = f(model);
        drop(guard);
        Ok(result)
    }
}

struct ExecutionGuard<'a> {
    handle: &'a LoadedModelHandle,
}

impl Drop for ExecutionGuard<'_> {
    fn drop(&mut self) {
        let mut lifecycle = self.handle.inner.state.lock().unwrap();
        lifecycle.state = if lifecycle.accepting_new_work {
            ModelState::Ready
        } else {
            ModelState::Closed
        };
        self.handle.inner.state_changed.notify_all();
    }
}

#[derive(Clone)]
pub struct ModelManager {
    registry: Arc<DriverRegistry>,
    next_instance_id: Arc<AtomicU64>,
    instances: Arc<Mutex<BTreeMap<u64, LoadedModelHandle>>>,
}

impl ModelManager {
    pub fn new(registry: DriverRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            instances: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn load<C>(&self, driver_id: &str, config: C) -> Result<LoadedModelHandle, DiagnosticError>
    where
        C: Any + Send + 'static,
    {
        let driver = self.registry.get(driver_id).ok_or_else(|| {
            DiagnosticError::not_found(
                "driver.not_registered",
                format!("driver `{driver_id}` is not registered"),
            )
        })?;
        let model = driver.load_boxed(Box::new(config))?;
        let instance_id = self.next_instance_id.fetch_add(1, Ordering::SeqCst);
        let handle = LoadedModelHandle::new(instance_id, model);
        self.instances
            .lock()
            .unwrap()
            .insert(instance_id, handle.clone());
        Ok(handle)
    }

    pub fn get(&self, instance_id: u64) -> Option<LoadedModelHandle> {
        self.instances.lock().unwrap().get(&instance_id).cloned()
    }

    pub fn remove(&self, instance_id: u64) -> Result<LoadedModelHandle, DiagnosticError> {
        let mut instances = self.instances.lock().unwrap();
        let handle = instances.get(&instance_id).cloned().ok_or_else(|| {
            DiagnosticError::not_found(
                "model.not_found",
                format!("model instance {instance_id} is not tracked"),
            )
        })?;

        if handle.state() != ModelState::Closed {
            return Err(DiagnosticError::conflict(
                "model.remove_requires_closed",
                format!("model instance {instance_id} must be closed before removal"),
            ));
        }

        Ok(instances.remove(&instance_id).unwrap())
    }
}
