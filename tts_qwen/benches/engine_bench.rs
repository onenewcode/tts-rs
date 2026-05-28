mod common;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tts_core::{ComputeBackend, ModelRegistry, SynthesisOptions, SynthesisRequest, TtsService};
use tts_qwen::{available_backends, register_qwen_family_model};

const MODEL_ID: &str = "qwen-bench";
const VARIANT: &str = "qwen3-tts-12hz-0.6b-customvoice";

fn bench_engine(c: &mut Criterion) {
    let Some(model_dir) = common::require_model_dir("engine_bench") else {
        return;
    };

    let backends = available_backends();
    if backends.is_empty() {
        eprintln!("skipping engine_bench: no runtime backend feature is enabled");
        return;
    }

    let mut group = c.benchmark_group("service_synthesize");
    for backend in backends {
        for case in common::synthetic_request_cases() {
            let backend_label = backend.label().to_string();
            group.bench_with_input(
                BenchmarkId::new(backend_label, case.name),
                &case,
                |b, case| {
                    b.iter_batched(
                        || build_service(&model_dir),
                        |service| {
                            service
                                .synthesize(
                                    MODEL_ID,
                                    &SynthesisRequest {
                                        text: case.text.to_string(),
                                        language: case.language.map(str::to_string),
                                        speaker: case.speaker.map(str::to_string),
                                    },
                                    &SynthesisOptions {
                                        max_new_tokens: 24,
                                        backend: Some(map_backend(backend)),
                                        ..SynthesisOptions::default()
                                    },
                                )
                                .expect("synthesis should succeed")
                        },
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
    group.finish();
}

fn build_service(model_dir: &std::path::Path) -> TtsService {
    let mut registry = ModelRegistry::new();
    assert!(register_qwen_family_model(
        &mut registry,
        MODEL_ID,
        model_dir,
        VARIANT,
    ));
    TtsService::new(registry)
}

fn map_backend(backend: tts_qwen::BackendKind) -> ComputeBackend {
    match backend {
        tts_qwen::BackendKind::Flex => ComputeBackend::Flex,
        tts_qwen::BackendKind::Wgpu => ComputeBackend::Wgpu,
        tts_qwen::BackendKind::Cuda => ComputeBackend::Cuda,
        tts_qwen::BackendKind::Rocm => ComputeBackend::Rocm,
        tts_qwen::BackendKind::Metal => ComputeBackend::Metal,
        tts_qwen::BackendKind::Vulkan => ComputeBackend::Vulkan,
        tts_qwen::BackendKind::WebGpu => ComputeBackend::WebGpu,
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_engine
}
criterion_main!(benches);
