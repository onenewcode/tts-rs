mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use tts_core::runtime::sampling::{SamplingConfig, sample_token};

#[cfg(feature = "flex")]
fn bench_sampling(c: &mut Criterion) {
    type BenchBackend = burn::backend::Flex;

    let device = Default::default();
    let logits = common::synthetic_logits::<BenchBackend>(&device, 1, 1, 4096);

    let mut group = c.benchmark_group("sample_token");
    for (name, config) in [
        ("greedy", SamplingConfig::greedy()),
        (
            "top_k",
            SamplingConfig {
                do_sample: true,
                temperature: 0.8,
                top_k: Some(64),
                top_p: 1.0,
                seed: None,
                repetition_penalty: None,
            },
        ),
        (
            "top_p",
            SamplingConfig {
                do_sample: true,
                temperature: 0.8,
                top_k: None,
                top_p: 0.9,
                seed: None,
                repetition_penalty: None,
            },
        ),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| sample_token::<BenchBackend>(logits.clone(), &config, None, &[], &device));
        });
    }
    group.finish();
}

#[cfg(not(feature = "flex"))]
fn bench_sampling(_: &mut Criterion) {
    eprintln!("skipping sampling_bench: requires feature `flex`");
}

criterion_group!(benches, bench_sampling);
criterion_main!(benches);
