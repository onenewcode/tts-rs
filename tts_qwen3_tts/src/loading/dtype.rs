use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::{Backend, DeviceError, get_device_settings, set_default_float_dtype};
use burn::tensor::quantization::{QTensorPrimitive, QuantLevel};
use burn::tensor::{DType, Tensor};

use crate::{Qwen3TtsLoadError, Qwen3TtsModelDType};

pub(crate) fn initialize_device_dtype<B: Backend>(
    device: &B::Device,
    dtype: Option<Qwen3TtsModelDType>,
) -> Result<(), Qwen3TtsLoadError> {
    let dtype = Qwen3TtsModelDType::resolve(dtype)
        .float_dtype()
        .unwrap_or(burn::tensor::FloatDType::BF16);

    match set_default_float_dtype::<B>(device, dtype) {
        Ok(()) => Ok(()),
        Err(DeviceError::AlreadyInitialized { .. })
            if get_device_settings::<B>(device).float_dtype == dtype =>
        {
            Ok(())
        }
        Err(source) => Err(Qwen3TtsLoadError::RuntimeDType {
            requested: DType::from(dtype).name().to_string(),
            message: source.to_string(),
        }),
    }
}

pub(crate) fn convert_module_dtype<B, M>(module: M, dtype: Option<Qwen3TtsModelDType>) -> M
where
    B: Backend,
    M: Module<B>,
{
    let dtype = Qwen3TtsModelDType::resolve(dtype);
    let mut mapper = RuntimeDTypeMapper { dtype };
    module.map(&mut mapper)
}

pub(crate) fn quantize_talker_linears<B, M>(module: M, dtype: Option<Qwen3TtsModelDType>) -> M
where
    B: Backend,
    M: Module<B>,
{
    let Some(value) = Qwen3TtsModelDType::resolve(dtype).quant_value() else {
        return module;
    };
    let scheme = <B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
        .with_value(value)
        .with_level(QuantLevel::Tensor);
    let mut mapper = TalkerLinearQuantizationMapper {
        scheme,
        module_stack: Vec::new(),
    };
    module.map(&mut mapper)
}

struct RuntimeDTypeMapper {
    dtype: Qwen3TtsModelDType,
}

impl<B: Backend> ModuleMapper<B> for RuntimeDTypeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let dtype = self.dtype;
        let target = dtype.float_dtype().map(DType::from).unwrap_or(DType::BF16);
        param.map(|tensor| tensor.dequantize().cast(target))
    }
}

struct TalkerLinearQuantizationMapper {
    scheme: burn::tensor::quantization::QuantScheme,
    module_stack: Vec<(String, String)>,
}

impl TalkerLinearQuantizationMapper {
    fn in_linear_module(&self) -> bool {
        self.module_stack
            .last()
            .is_some_and(|(_, container_type)| container_type.contains("Linear"))
    }
}

impl<B: Backend> ModuleMapper<B> for TalkerLinearQuantizationMapper {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.module_stack
            .push((name.to_string(), container_type.to_string()));
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        let _ = self.module_stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D == 2 && self.in_linear_module() {
            let scheme = self.scheme;
            param.map(|tensor| tensor.dequantize().quantize_dynamic(&scheme))
        } else {
            let target = DType::BF16;
            param.map(|tensor| tensor.dequantize().cast(target))
        }
    }
}
