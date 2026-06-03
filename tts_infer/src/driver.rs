use std::any::Any;
use std::sync::Arc;

use tts_error::DiagnosticError;

use crate::{DriverDescriptor, ModelCapabilities};

pub trait ErasedLoadedModel: Any + Send + Sync {
    fn driver_id(&self) -> &'static str;
    fn capabilities(&self) -> ModelCapabilities;
    fn as_any(&self) -> &dyn Any;
}

pub trait DriverFactory: Send + Sync + 'static {
    type Config: Any + Send + 'static;

    fn descriptor(&self) -> DriverDescriptor;
    fn load(&self, config: Self::Config) -> Result<Box<dyn ErasedLoadedModel>, DiagnosticError>;
}

pub(crate) trait ErasedDriverFactory: Send + Sync {
    fn descriptor(&self) -> DriverDescriptor;
    fn load_boxed(
        &self,
        config: Box<dyn Any + Send>,
    ) -> Result<Box<dyn ErasedLoadedModel>, DiagnosticError>;
}

impl<T> ErasedDriverFactory for T
where
    T: DriverFactory,
{
    fn descriptor(&self) -> DriverDescriptor {
        DriverFactory::descriptor(self)
    }

    fn load_boxed(
        &self,
        config: Box<dyn Any + Send>,
    ) -> Result<Box<dyn ErasedLoadedModel>, DiagnosticError> {
        let config = config.downcast::<T::Config>().map_err(|_| {
            DiagnosticError::invalid_argument(
                "driver.config_type_mismatch",
                format!(
                    "driver `{}` expected config type `{}`",
                    self.descriptor().driver_id(),
                    self.descriptor().config_type(),
                ),
            )
        })?;
        self.load(*config)
    }
}

pub(crate) type SharedDriverFactory = Arc<dyn ErasedDriverFactory>;
