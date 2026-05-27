mod common;

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor};
use tts_qwen::{
    CustomVoiceRequest, Qwen3TtsPipeline, build_custom_voice_prompt,
    load_custom_voice_generation_config,
};

type Backend = Flex;

const SAMPLE_TEXT_IDS: &[i64] = &[
    151644, 77091, 198, 101045, 110146, 18830, 99879, 3837, 35946, 101909, 100654, 106614, 104144,
    101106, 104405, 100623, 1773, 151645, 198, 151644, 77091, 198,
];

#[test]
fn custom_voice_prompt_matches_qwen_tts_template() {
    let request = CustomVoiceRequest::new("hello");
    assert_eq!(
        build_custom_voice_prompt(&request),
        "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn generation_config_uses_codec_eos_and_suppresses_reserved_range() {
    let config = load_custom_voice_generation_config(&common::resolve_model_dir()).unwrap();
    assert_eq!(config.codec_eos_token_id, 2150);
    assert!(config.suppress_token_ids.contains(&2148));
    assert!(config.suppress_token_ids.contains(&2149));
    assert!(!config.suppress_token_ids.contains(&2150));
}

#[test]
fn pipeline_frontend_builds_expected_single_sample_controls_and_masks() {
    let model_dir = common::resolve_model_dir();
    let device = Default::default();
    let pipeline = Qwen3TtsPipeline::<Backend>::load(&model_dir, &device).unwrap();
    let request = CustomVoiceRequest {
        text: "其实我真的有发现，我是一个特别善于观察别人情绪的人。".to_string(),
        language: Some("Chinese".to_string()),
        speaker: Some("Vivian".to_string()),
    };

    let frontend = pipeline.build_frontend(&request).unwrap();

    assert_eq!(frontend.text_token_ids, vec![SAMPLE_TEXT_IDS.to_vec()]);
    assert_eq!(
        frontend.codec_prefix_ids,
        vec![vec![2154, 2156, 2055, 2157, 3065, 2148, 2149]]
    );
    let hidden_size = pipeline.talker_config().hidden_size;
    assert_eq!(frontend.inputs_embeds.dims(), [1, 25, hidden_size]);
    assert_eq!(frontend.trailing_text_hidden.dims(), [1, 1, hidden_size]);
    assert_eq!(frontend.tts_pad_embed.dims(), [1, 1, hidden_size]);
    assert_eq!(frontend.inputs_embeds.dtype(), DType::BF16);
    assert_eq!(frontend.trailing_text_hidden.dtype(), DType::BF16);
    assert_eq!(frontend.tts_pad_embed.dtype(), DType::BF16);
    assert_eq!(tensor_i32_2d(frontend.attention_mask), vec![1; 25]);
    assert_eq!(
        tensor_i32_3d(frontend.position_ids),
        (0..3).flat_map(|_| 0..25).collect::<Vec<i32>>()
    );
}

#[test]
fn pipeline_frontend_left_pads_batch_attention_and_positions() {
    let model_dir = common::resolve_model_dir();
    let device = Default::default();
    let pipeline = Qwen3TtsPipeline::<Backend>::load(&model_dir, &device).unwrap();
    let batch = tts_qwen::CustomVoiceBatch {
        requests: vec![
            CustomVoiceRequest {
                text: "hello".to_string(),
                language: Some("English".to_string()),
                speaker: None,
            },
            CustomVoiceRequest {
                text: "其实我真的有发现，我是一个特别善于观察别人情绪的人。".to_string(),
                language: Some("Chinese".to_string()),
                speaker: Some("Vivian".to_string()),
            },
        ],
    };

    let frontend = pipeline.build_frontend_batch(&batch).unwrap();

    assert_eq!(
        frontend.codec_prefix_ids[0],
        vec![2154, 2156, 2050, 2157, 2148, 2149]
    );
    assert_eq!(
        frontend.codec_prefix_ids[1],
        vec![2154, 2156, 2055, 2157, 3065, 2148, 2149]
    );
    let [batch_size, max_len, hidden] = frontend.inputs_embeds.dims();
    assert_eq!(
        [batch_size, hidden],
        [2, pipeline.talker_config().hidden_size]
    );

    let expected_lens = frontend
        .text_token_ids
        .iter()
        .zip(frontend.codec_prefix_ids.iter())
        .map(|(text_ids, prefix_ids)| text_ids.len() + prefix_ids.len() - 4)
        .collect::<Vec<_>>();
    assert_eq!(max_len, *expected_lens.iter().max().unwrap());

    let attention = tensor_i32_2d(frontend.attention_mask);
    for (sample_idx, seq_len) in expected_lens.iter().copied().enumerate() {
        let row = &attention[sample_idx * max_len..(sample_idx + 1) * max_len];
        let pad_len = max_len - seq_len;
        assert_eq!(&row[..pad_len], vec![0; pad_len].as_slice());
        assert_eq!(&row[pad_len..], vec![1; seq_len].as_slice());
    }

    let positions = tensor_i32_3d(frontend.position_ids);
    for axis in 0..3 {
        for (sample_idx, seq_len) in expected_lens.iter().copied().enumerate() {
            let start = axis * batch_size * max_len + sample_idx * max_len;
            let row = &positions[start..start + max_len];
            let pad_len = max_len - seq_len;
            assert_eq!(&row[..pad_len], vec![0; pad_len].as_slice());
            assert_eq!(
                &row[pad_len..],
                (0..seq_len as i32).collect::<Vec<_>>().as_slice()
            );
        }
    }
}

fn tensor_i32_2d(tensor: Tensor<Backend, 2, Int>) -> Vec<i32> {
    tensor
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap()
}

fn tensor_i32_3d(tensor: Tensor<Backend, 3, Int>) -> Vec<i32> {
    tensor
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .unwrap()
}
