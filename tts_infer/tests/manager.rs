use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use tts_core::driver::{DriverFactory, ErasedLoadedModel};
use tts_core::{DriverDescriptor, DriverRegistry, ModelCapabilities, ModelManager, ModelState};

#[derive(Debug, Clone)]
struct MockLoadConfig {
    support_voice_clone: bool,
}

#[derive(Debug)]
struct MockDriver;

impl DriverFactory for MockDriver {
    type Config = MockLoadConfig;

    fn descriptor(&self) -> DriverDescriptor {
        DriverDescriptor::new(
            "mock",
            "Mock Driver",
            "test double driver",
            std::any::type_name::<MockLoadConfig>(),
        )
    }

    fn load(
        &self,
        config: Self::Config,
    ) -> Result<Box<dyn ErasedLoadedModel>, tts_error::DiagnosticError> {
        Ok(Box::new(MockLoadedModel {
            running: Arc::new(AtomicUsize::new(0)),
            support_voice_clone: config.support_voice_clone,
        }))
    }
}

#[derive(Debug)]
struct MockLoadedModel {
    running: Arc<AtomicUsize>,
    support_voice_clone: bool,
}

impl ErasedLoadedModel for MockLoadedModel {
    fn driver_id(&self) -> &'static str {
        "mock"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::builder()
            .supports_voice_clone(self.support_voice_clone)
            .sample_rate_hz(24_000)
            .channels(1)
            .build()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn registry_rejects_duplicate_driver_ids_and_lists_descriptors_in_order() {
    let mut registry = DriverRegistry::new();
    registry.register(MockDriver).unwrap();

    let duplicate = registry.register(MockDriver).unwrap_err();
    assert!(duplicate.to_string().contains("mock"));

    let descriptors = registry.descriptors();
    assert_eq!(descriptors.len(), 1);
    assert_eq!(descriptors[0].driver_id(), "mock");
    assert_eq!(descriptors[0].display_name(), "Mock Driver");
}

#[test]
fn manager_loads_instances_assigns_ids_and_requires_close_before_remove() {
    let mut registry = DriverRegistry::new();
    registry.register(MockDriver).unwrap();
    let manager = ModelManager::new(registry);

    let handle = manager
        .load(
            "mock",
            MockLoadConfig {
                support_voice_clone: true,
            },
        )
        .unwrap();

    assert_eq!(handle.instance_id(), 1);
    assert_eq!(handle.driver_id(), "mock");
    assert_eq!(handle.state(), ModelState::Ready);
    assert!(handle.capabilities().supports_voice_clone);

    let error = manager.remove(handle.instance_id()).unwrap_err();
    assert!(error.to_string().contains("closed"));

    handle.close().unwrap();
    manager.remove(handle.instance_id()).unwrap();
    assert!(manager.get(handle.instance_id()).is_none());
}

#[test]
fn handle_serializes_work_and_close_waits_for_in_flight_execution() {
    let mut registry = DriverRegistry::new();
    registry.register(MockDriver).unwrap();
    let manager = ModelManager::new(registry);
    let handle = manager
        .load(
            "mock",
            MockLoadConfig {
                support_voice_clone: false,
            },
        )
        .unwrap();

    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let worker_handle = handle.clone();
    let entered_worker = entered.clone();
    let release_worker = release.clone();

    let join = thread::spawn(move || {
        worker_handle
            .with_model_as::<MockLoadedModel, _, _>(|model| {
                assert_eq!(model.running.fetch_add(1, Ordering::SeqCst), 0);
                entered_worker.wait();
                release_worker.wait();
                thread::sleep(Duration::from_millis(20));
                model.running.fetch_sub(1, Ordering::SeqCst);
            })
            .unwrap();
    });

    entered.wait();
    assert_eq!(handle.state(), ModelState::Busy);

    let closer = thread::spawn({
        let handle = handle.clone();
        move || handle.close().unwrap()
    });

    thread::sleep(Duration::from_millis(20));
    assert_eq!(handle.state(), ModelState::Busy);

    release.wait();
    join.join().unwrap();
    closer.join().unwrap();

    assert_eq!(handle.state(), ModelState::Closed);
    let error = handle
        .with_model_as::<MockLoadedModel, _, _>(|_| ())
        .unwrap_err();
    assert!(error.to_string().contains("closed"));
}
