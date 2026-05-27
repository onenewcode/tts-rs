mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use tts_qwen::{build_custom_voice_prompt, load_qwen3_tts_tokenizer};

fn bench_tokenizer(c: &mut Criterion) {
    let Some(model_dir) = common::require_model_dir("tokenizer_bench") else {
        return;
    };

    let mut load_group = c.benchmark_group("tokenizer_load");
    load_group.bench_function("qwen3_tts", |b| {
        b.iter(|| load_qwen3_tts_tokenizer(&model_dir).expect("tokenizer should load"));
    });
    load_group.finish();

    let tokenizer = load_qwen3_tts_tokenizer(&model_dir).expect("tokenizer should load");
    let mut encode_group = c.benchmark_group("tokenizer_encode");
    for case in common::synthetic_request_cases() {
        let prompt = build_custom_voice_prompt(&case.build_request());
        encode_group.bench_function(case.name, |b| {
            b.iter(|| {
                tokenizer
                    .encode(prompt.as_str(), false)
                    .expect("encode should succeed")
            });
        });
    }
    encode_group.finish();
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
