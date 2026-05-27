mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use tts_qwen::{CustomVoiceRequest, build_custom_voice_prompt};

fn bench_prompt_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_build");
    for case in common::synthetic_request_cases() {
        let request = CustomVoiceRequest::new(case.text);
        group.bench_function(case.name, |b| {
            b.iter(|| build_custom_voice_prompt(&request));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prompt_build);
criterion_main!(benches);
