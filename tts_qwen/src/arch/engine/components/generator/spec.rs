use crate::arch::engine::spec::{ComponentKind, ComponentSpec, QWEN_ENGINE_SPEC};

pub(crate) fn generator_component_spec() -> &'static ComponentSpec {
    QWEN_ENGINE_SPEC
        .component(ComponentKind::AcousticGenerator)
        .expect("generator component spec should exist")
}
