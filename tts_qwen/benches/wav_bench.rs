mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use tts_qwen::write_pcm_wav;

fn bench_write_pcm_wav(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_pcm_wav");
    for (name, pcm) in [
        (
            "short_pcm",
            common::synthetic_pcm(common::SAMPLE_RATE as usize / 2),
        ),
        (
            "long_pcm",
            common::synthetic_pcm(common::SAMPLE_RATE as usize * 4),
        ),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(pcm.len() * 2 + 44);
                write_pcm_wav(&pcm, &mut buffer, common::SAMPLE_RATE)
                    .expect("wav write should succeed");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_write_pcm_wav);
criterion_main!(benches);
