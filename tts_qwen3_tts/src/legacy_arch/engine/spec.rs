#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnginePolicy {
    PrepareFirst,
    SequenceBoundary,
    AssemblyOverRegistry,
    EdgeDeviceResourcePriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentKind {
    AcousticGenerator,
    AudioDecoder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineFacts {
    pub(crate) dag_nodes: [&'static str; 2],
    pub(crate) dag_edges: [(&'static str, &'static str, &'static str); 1],
    pub(crate) protocols: [&'static str; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceContract {
    pub(crate) workspace: &'static str,
    pub(crate) session_model: &'static str,
    pub(crate) resource_policy: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionalContract {
    pub(crate) input: &'static str,
    pub(crate) output: &'static str,
    pub(crate) description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutionBoundaryContract {
    pub(crate) accepts: &'static str,
    pub(crate) executes_on: &'static str,
    pub(crate) produces: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComponentSpec {
    pub(crate) kind: ComponentKind,
    pub(crate) label: &'static str,
    pub(crate) functional: FunctionalContract,
    pub(crate) resource_contract: ResourceContract,
    pub(crate) execution_boundary: ExecutionBoundaryContract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineSpec {
    pub(crate) name: &'static str,
    pub(crate) facts: EngineFacts,
    pub(crate) policies: &'static [EnginePolicy],
    pub(crate) resource_contract: ResourceContract,
    pub(crate) components: &'static [ComponentSpec],
}

impl EngineSpec {
    pub(crate) fn component(&self, kind: ComponentKind) -> Option<&ComponentSpec> {
        self.components.iter().find(|spec| spec.kind == kind)
    }
}

const ENGINE_POLICIES: &[EnginePolicy] = &[
    EnginePolicy::PrepareFirst,
    EnginePolicy::SequenceBoundary,
    EnginePolicy::AssemblyOverRegistry,
    EnginePolicy::EdgeDeviceResourcePriority,
];

const ENGINE_COMPONENTS: &[ComponentSpec] = &[
    ComponentSpec {
        kind: ComponentKind::AcousticGenerator,
        label: "acoustic_generator",
        functional: FunctionalContract {
            input: "PreparedCondition",
            output: "CodecTokenSequence",
            description: "Decode prepared linguistic and control conditions into full codec-token sequences.",
        },
        resource_contract: ResourceContract {
            workspace: "fixed_capacity",
            session_model: "single_request",
            resource_policy: "edge_device_priority",
        },
        execution_boundary: ExecutionBoundaryContract {
            accepts: "PreparedCondition",
            executes_on: "GeneratorExecutionForm",
            produces: "CodecTokenSequence",
        },
    },
    ComponentSpec {
        kind: ComponentKind::AudioDecoder,
        label: "audio_decoder",
        functional: FunctionalContract {
            input: "CodecTokenSequence",
            output: "Waveform",
            description: "Decode finalized codec-token sequences into waveform tensors.",
        },
        resource_contract: ResourceContract {
            workspace: "fixed_capacity",
            session_model: "single_request",
            resource_policy: "edge_device_priority",
        },
        execution_boundary: ExecutionBoundaryContract {
            accepts: "CodecTokenSequence",
            executes_on: "DecoderExecutionForm",
            produces: "Waveform",
        },
    },
];

pub(crate) const fn qwen_engine_spec() -> EngineSpec {
    EngineSpec {
        name: "qwen_edge_engine",
        facts: EngineFacts {
            dag_nodes: ["acoustic_generator", "audio_decoder"],
            dag_edges: [("acoustic_generator", "audio_decoder", "CodecTokenSequence")],
            protocols: ["PreparedCondition", "CodecTokenSequence", "Waveform"],
        },
        policies: ENGINE_POLICIES,
        resource_contract: ResourceContract {
            workspace: "fixed_capacity",
            session_model: "single_request",
            resource_policy: "spec_modeled",
        },
        components: ENGINE_COMPONENTS,
    }
}


pub(crate) static QWEN_ENGINE_SPEC: EngineSpec = qwen_engine_spec();
