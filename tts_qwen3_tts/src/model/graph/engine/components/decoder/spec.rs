use crate::model::graph::engine::spec::{ComponentKind, ComponentSpec, QWEN_ENGINE_SPEC};

pub(crate) fn decoder_component_spec() -> &'static ComponentSpec {
    QWEN_ENGINE_SPEC
        .component(ComponentKind::AudioDecoder)
        .expect("decoder component spec should exist")
}
