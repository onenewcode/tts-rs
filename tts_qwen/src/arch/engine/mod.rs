pub mod assembly;
pub mod compiler;
pub mod components;
pub mod protocol;
pub mod spec;

#[cfg(test)]
mod tests {
    use burn::backend::Flex;

    use super::components::{
        decoder::lowering::DecoderLowering,
        generator::{
            import::config::{
                Qwen3TtsConfig, Qwen3TtsTalkerCodePredictorConfig, Qwen3TtsTalkerConfig,
                Qwen3TtsTalkerRopeScalingConfig,
            },
            lowering::GeneratorLowering,
            weights::LoadedQwen3TtsTalker,
        },
    };
    use super::protocol::{CodecTokenSequence, PreparedCondition};
    use super::spec::{ComponentKind, EnginePolicy, qwen_engine_spec};
    use crate::profile::compile::SemanticRequestCondition;
    use crate::profile::model_config::ProfileControlIds;

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
        let talker = synthetic_loaded_talker::<TestBackend>(&device);
        let prepared = PreparedCondition::new(
            "qwen3-tts-12hz-0.6b-base",
            SemanticRequestCondition {
                text_token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                controls: ProfileControlIds {
                    tts_bos_token_id: 1,
                    tts_eos_token_id: 2,
                    tts_pad_token_id: 3,
                    codec_bos_id: 4,
                    codec_pad_id: 5,
                    codec_prefix_ids: vec![4, 6, 7],
                },
            },
        )
        .unwrap();

        assert_eq!(prepared.sequence_len_hint(), 12);

        let execution = GeneratorLowering::lower(
            &prepared,
            &talker.config.talker_config,
            &talker,
            &device,
        )
        .unwrap();
        assert_eq!(execution.batch_size(), 1);
        assert!(execution.sequence_len() > 0);
        assert_eq!(prepared.release_label(), "qwen3-tts-12hz-0.6b-base");
    }

    #[test]
    fn decoder_lowering_requires_complete_codec_token_sequences() {
        let device = Default::default();
        let sequence = CodecTokenSequence::new(vec![1, 2, 3, 4], 1, 2, 2).unwrap();
        let execution = DecoderLowering::lower::<TestBackend>(&sequence, &device).unwrap();
        assert_eq!(execution.batch_size(), 1);
        assert_eq!(execution.num_quantizers(), 2);
        assert_eq!(execution.time_steps(), 2);

        let error = CodecTokenSequence::new(vec![], 1, 2, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("time dimension must be non-zero"));
    }

    #[test]
    fn waveform_protocol_keeps_semantic_shape_metadata() {
        let waveform = super::protocol::Waveform::new(24_000, 1, 1, vec![0.0, 0.5, -0.5]).unwrap();
        assert_eq!(waveform.sample_rate(), 24_000);
        assert_eq!(waveform.batch_size(), 1);
        assert_eq!(waveform.channels(), 1);
        assert_eq!(waveform.samples().len(), 3);
    }

    fn synthetic_loaded_talker<B: burn::tensor::backend::Backend>(
        device: &B::Device,
    ) -> LoadedQwen3TtsTalker<B> {
        let config = Qwen3TtsConfig {
            talker_config: Qwen3TtsTalkerConfig {
                code_predictor_config: Qwen3TtsTalkerCodePredictorConfig {
                    vocab_size: 16,
                    hidden_size: 4,
                    intermediate_size: 8,
                    hidden_act: "silu".to_string(),
                    num_hidden_layers: 1,
                    num_attention_heads: 1,
                    num_key_value_heads: 1,
                    head_dim: 4,
                    max_position_embeddings: 32,
                    rms_norm_eps: 1e-6,
                    rope_theta: 10_000.0,
                    attention_bias: false,
                    num_code_groups: 3,
                },
                vocab_size: 16,
                hidden_size: 4,
                intermediate_size: 8,
                hidden_act: "silu".to_string(),
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 32,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                rope_scaling: Qwen3TtsTalkerRopeScalingConfig::default(),
                attention_bias: false,
                num_code_groups: 3,
                text_hidden_size: 4,
                text_vocab_size: 32,
            },
        };
        let model = config.init_checkpoint(device);
        LoadedQwen3TtsTalker { config, model }
    }
}
