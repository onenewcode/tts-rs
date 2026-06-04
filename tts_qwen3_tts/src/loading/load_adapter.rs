use std::rc::Rc;

use burn::tensor::DType;
use burn_store::{ModuleAdapter, TensorSnapshot};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoadTimeFloatAdapter {
    target_dtype: DType,
}

impl LoadTimeFloatAdapter {
    pub(crate) fn new(target_dtype: DType) -> Self {
        Self { target_dtype }
    }
}

impl ModuleAdapter for LoadTimeFloatAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if !matches!(
            snapshot.dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
        ) {
            return snapshot.clone();
        }

        if snapshot.dtype == self.target_dtype {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();
        let target_dtype = self.target_dtype;
        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(target_dtype))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            target_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(*self)
    }
}
