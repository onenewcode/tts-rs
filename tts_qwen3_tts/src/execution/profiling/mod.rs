mod config;

pub use config::Qwen3TtsProfilingConfig;

#[cfg(feature = "operator-profiling")]
use std::cell::RefCell;
#[cfg(feature = "operator-profiling")]
use std::sync::{OnceLock, RwLock};
#[cfg(feature = "operator-profiling")]
use std::time::Instant;

#[cfg(feature = "operator-profiling")]
static PROFILING_CONFIG: OnceLock<RwLock<Qwen3TtsProfilingConfig>> = OnceLock::new();

#[cfg(feature = "operator-profiling")]
thread_local! {
    static CONTEXT: RefCell<Option<ProfilingContext>> = const { RefCell::new(None) };
}

#[cfg(feature = "operator-profiling")]
#[derive(Clone, Copy)]
struct ProfilingContext {
    session_id: usize,
    step_idx: usize,
}

pub(crate) fn configure(config: &Qwen3TtsProfilingConfig) {
    #[cfg(feature = "operator-profiling")]
    {
        let lock = PROFILING_CONFIG.get_or_init(|| RwLock::new(Qwen3TtsProfilingConfig::default()));
        *lock.write().expect("profiling config lock poisoned") = config.clone();
    }
    #[cfg(not(feature = "operator-profiling"))]
    let _ = config;
}

pub(crate) fn with_session_context<T>(
    session_id: usize,
    step_idx: usize,
    f: impl FnOnce() -> T,
) -> T {
    #[cfg(feature = "operator-profiling")]
    {
        CONTEXT.with(|slot| {
            let previous = slot.replace(Some(ProfilingContext {
                session_id,
                step_idx,
            }));
            let output = f();
            slot.replace(previous);
            output
        })
    }
    #[cfg(not(feature = "operator-profiling"))]
    {
        let _ = (session_id, step_idx);
        f()
    }
}

pub(crate) fn record_operator<T>(_name: &'static str, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "operator-profiling")]
    {
        let enabled = PROFILING_CONFIG
            .get_or_init(|| RwLock::new(Qwen3TtsProfilingConfig::default()))
            .read()
            .expect("profiling config lock poisoned")
            .enabled;
        if enabled {
            let started = Instant::now();
            let output = f();
            let elapsed_us = started.elapsed().as_micros();
            CONTEXT.with(|slot| {
                if let Some(context) = *slot.borrow() {
                    tracing::info!(
                        session_id = context.session_id,
                        step_idx = context.step_idx,
                        op = _name,
                        elapsed_us,
                        "operator_timing"
                    );
                } else {
                    tracing::info!(op = _name, elapsed_us, "operator_timing");
                }
            });
            return output;
        }
    }
    f()
}
