mod common;

use std::path::Path;

use burn::tensor::backend::Backend;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use tts_qwen::{QwenTtsEngine, default_session_config};

fn bench_engine(c: &mut Criterion) {
    let Some(model_dir) = common::require_model_dir("engine_bench") else {
        return;
    };

    #[cfg(feature = "flex")]
    {
        let device = Default::default();
        bench_backend::<burn::backend::Flex>(c, &model_dir, "flex", device);
    }

    #[cfg(feature = "wgpu")]
    {
        let device = Default::default();
        bench_backend::<burn::backend::Wgpu>(c, &model_dir, "wgpu", device);
    }

    #[cfg(feature = "cuda")]
    {
        let device = Default::default();
        bench_backend::<burn::backend::Cuda>(c, &model_dir, "cuda", device);
    }

    #[cfg(feature = "rocm")]
    {
        let device = Default::default();
        bench_backend::<burn::backend::Rocm>(c, &model_dir, "rocm", device);
    }

    #[cfg(feature = "metal")]
    {
        let device = Default::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &device,
            Default::default(),
        );
        bench_backend::<burn::backend::Metal>(c, &model_dir, "metal", device);
    }

    #[cfg(feature = "vulkan")]
    {
        let device = Default::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            Default::default(),
        );
        bench_backend::<burn::backend::Vulkan>(c, &model_dir, "vulkan", device);
    }

    #[cfg(feature = "webgpu")]
    {
        let device = Default::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::WebGpu>(
            &device,
            Default::default(),
        );
        bench_backend::<burn::backend::WebGpu>(c, &model_dir, "webgpu", device);
    }

    #[cfg(not(any(
        feature = "flex",
        feature = "wgpu",
        feature = "cuda",
        feature = "rocm",
        feature = "metal",
        feature = "vulkan",
        feature = "webgpu"
    )))]
    {
        eprintln!("skipping engine_bench: no runtime backend feature is enabled");
    }
}

fn bench_backend<B>(c: &mut Criterion, model_dir: &Path, backend_label: &str, device: B::Device)
where
    B: Backend,
    B::Device: Clone,
{
    bench_engine_load::<B>(c, model_dir, backend_label, &device);
    bench_first_step::<B>(c, model_dir, backend_label, &device);
    bench_run_to_end::<B>(c, model_dir, backend_label, &device);
}

fn bench_engine_load<B>(
    c: &mut Criterion,
    model_dir: &Path,
    backend_label: &str,
    device: &B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let mut group = c.benchmark_group("engine_load");
    group.bench_function(backend_label, |b| {
        b.iter(|| {
            QwenTtsEngine::<B>::load(model_dir, device, common::engine_config())
                .expect("engine load should succeed")
        });
    });
    group.finish();
}

fn bench_first_step<B>(c: &mut Criterion, model_dir: &Path, backend_label: &str, device: &B::Device)
where
    B: Backend,
    B::Device: Clone,
{
    let mut group = c.benchmark_group("session_first_step");
    bench_session_group::<B, _, _>(
        &mut group,
        model_dir,
        backend_label,
        device,
        |engine, request| {
            let handle = engine
                .start_session(request, common::session_config(24))
                .expect("session should start");
            engine.step(handle).expect("first step should succeed")
        },
    );
    group.finish();
}

fn bench_run_to_end<B>(c: &mut Criterion, model_dir: &Path, backend_label: &str, device: &B::Device)
where
    B: Backend,
    B::Device: Clone,
{
    let mut group = c.benchmark_group("session_run_to_end");
    bench_session_group::<B, _, _>(
        &mut group,
        model_dir,
        backend_label,
        device,
        |engine, request| {
            let handle = engine
                .start_session(request, default_session_config(24, false))
                .expect("session should start");
            engine
                .run_to_end(handle)
                .expect("run_to_end should succeed")
        },
    );
    group.finish();
}

fn bench_session_group<B, F, T>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    model_dir: &Path,
    backend_label: &str,
    device: &B::Device,
    mut routine: F,
) where
    B: Backend,
    B::Device: Clone,
    F: FnMut(&mut QwenTtsEngine<B>, tts_qwen::CustomVoiceRequest) -> T,
{
    for case in common::synthetic_request_cases() {
        group.bench_with_input(
            BenchmarkId::new(backend_label, case.name),
            &case,
            |b, case| {
                b.iter_batched(
                    || {
                        let engine =
                            QwenTtsEngine::<B>::load(model_dir, device, common::engine_config())
                                .expect("engine load should succeed");
                        (engine, case.build_request())
                    },
                    |(mut engine, request)| routine(&mut engine, request),
                    BatchSize::PerIteration,
                );
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_engine
}
criterion_main!(benches);
