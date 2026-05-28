pub mod assembly;
pub mod compiler;
pub mod components;
pub mod protocol;
pub mod spec;

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor, TensorData};

    use super::components::{
        decoder::lowering::DecoderLowering,
        generator::lowering::GeneratorLowering,
    };
    use super::protocol::{CodecTokenSequence, PreparedCondition};
    use super::spec::{ComponentKind, EnginePolicy, qwen_engine_spec};
    use crate::profile::compile::CompiledRequest;

    type TestBackend = Flex;

    #[test]
    fn engine_spec_expresses_prepare_first_sequence_boundary_and_resource_policy() {
        let spec = qwen_engine_spec();
        assert_eq!(spec.facts.dag_nodes, ["acoustic_generator", "audio_decoder"]);
        assert_eq!(spec.facts.protocols, ["PreparedCondition", "CodecTokenSequence", "Waveform"]);
        assert!(spec.policies.contains(&EnginePolicy::PrepareFirst));
        assert!(spec.policies.contains(&EnginePolicy::SequenceBoundary));
        assert!(spec.policies.contains(&EnginePolicy::AssemblyOverRegistry));
        assert!(spec.policies.contains(&EnginePolicy::EdgeDeviceResourcePriority));
        assert_eq!(spec.resource_contract.workspace, "fixed_capacity");
    }

    #[test]
    fn component_specs_capture_generator_and_decoder_boundaries() {
        let spec = qwen_engine_spec();
        let generator = spec.component(ComponentKind::AcousticGenerator).unwrap();
        assert_eq!(generator.execution_boundary.accepts, "PreparedCondition");
        assert_eq!(generator.execution_boundary.executes_on, "GeneratorExecutionForm");
        assert_eq!(generator.execution_boundary.produces, "CodecTokenSequence");

        let decoder = spec.component(ComponentKind::AudioDecoder).unwrap();
        assert_eq!(decoder.execution_boundary.accepts, "CodecTokenSequence");
        assert_eq!(decoder.execution_boundary.executes_on, "DecoderExecutionForm");
        assert_eq!(decoder.execution_boundary.produces, "Waveform");
    }

    #[test]
    fn generator_lowering_keeps_request_semantics_outside_graph() {
        let device = Default::default();
        let compiled = CompiledRequest {
            inputs_embeds: Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device),
            position_ids: Tensor::<TestBackend, 3, Int>::zeros([3, 1, 2], &device),
            attention_mask: Tensor::<TestBackend, 2, Int>::ones([1, 2], &device),
            trailing_text_hidden: Tensor::<TestBackend, 3>::zeros([1, 1, 4], &device),
            tts_pad_embed: Tensor::<TestBackend, 3>::zeros([1, 1, 4], &device),
        };
        let prepared = PreparedCondition::new("qwen3-tts-12hz-0.6b-base", compiled).unwrap();

        let execution = GeneratorLowering::lower(prepared).unwrap();
        assert_eq!(execution.batch_size(), 1);
        assert_eq!(execution.sequence_len(), 2);
        assert_eq!(execution.into_prepared().release_label(), "qwen3-tts-12hz-0.6b-base");
    }

    #[test]
    fn decoder_lowering_requires_complete_codec_token_sequences() {
        let device = Default::default();
        let tokens = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::new(vec![1, 2, 3, 4], [1, 2, 2]),
            &device,
        );
        let sequence = CodecTokenSequence::new(tokens, 2).unwrap();
        let execution = DecoderLowering::lower(sequence).unwrap();
        assert_eq!(execution.batch_size(), 1);
        assert_eq!(execution.num_quantizers(), 2);
        assert_eq!(execution.time_steps(), 2);

        let empty = Tensor::<TestBackend, 3, Int>::zeros([1, 2, 0], &device);
        let error = CodecTokenSequence::new(empty, 2).unwrap_err().to_string();
        assert!(error.contains("time dimension must be non-zero"));
    }
}
