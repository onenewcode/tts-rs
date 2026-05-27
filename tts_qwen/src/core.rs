use std::marker::PhantomData;
use std::path::Path;
use std::time::Instant;

use burn::tensor::backend::Backend;

use crate::SamplingConfig;

#[derive(Debug, Clone)]
pub struct LocalInferenceOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

impl Default for LocalInferenceOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::greedy(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalInferenceStageProfile {
    pub name: &'static str,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalInferenceProfile {
    pub total_elapsed_ms: u128,
    pub stages: Vec<LocalInferenceStageProfile>,
}

impl LocalInferenceProfile {
    pub fn push_stage(&mut self, name: &'static str, elapsed_ms: u128) {
        self.stages
            .push(LocalInferenceStageProfile { name, elapsed_ms });
    }

    pub fn record<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let output = f();
        self.push_stage(name, started.elapsed().as_millis());
        output
    }

    pub fn record_result<T, E>(
        &mut self,
        name: &'static str,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let started = Instant::now();
        let output = f();
        self.push_stage(name, started.elapsed().as_millis());
        output
    }
}

#[derive(Debug)]
pub struct LocalInferenceRun<T> {
    pub output: T,
    pub profile: LocalInferenceProfile,
}

pub trait LocalModelAdapter<B>: Sized
where
    B: Backend,
    B::Device: Clone,
{
    type Request;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;
    type LoadReport: Clone + std::fmt::Debug;

    fn load(model_dir: &Path, device: &B::Device) -> Result<Self, Self::Error>;

    fn load_report(&self) -> Self::LoadReport;

    fn infer(
        &self,
        request: &Self::Request,
        options: &LocalInferenceOptions,
        profile: &mut LocalInferenceProfile,
    ) -> Result<Self::Output, Self::Error>;

    fn write_output(&self, output: &Self::Output, path: &Path) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct LocalInferenceCore<B, A>
where
    B: Backend,
    B::Device: Clone,
    A: LocalModelAdapter<B>,
{
    adapter: A,
    _backend: PhantomData<B>,
}

impl<B, A> LocalInferenceCore<B, A>
where
    B: Backend,
    B::Device: Clone,
    A: LocalModelAdapter<B>,
{
    pub fn load(model_dir: impl AsRef<Path>, device: &B::Device) -> Result<Self, A::Error> {
        let adapter = A::load(model_dir.as_ref(), device)?;
        Ok(Self::new(adapter))
    }

    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
            _backend: PhantomData,
        }
    }

    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    pub fn load_report(&self) -> A::LoadReport {
        self.adapter.load_report()
    }

    pub fn infer(
        &self,
        request: &A::Request,
        options: &LocalInferenceOptions,
    ) -> Result<LocalInferenceRun<A::Output>, A::Error> {
        let started = Instant::now();
        let mut profile = LocalInferenceProfile::default();
        let output = self.adapter.infer(request, options, &mut profile)?;
        profile.total_elapsed_ms = started.elapsed().as_millis();
        Ok(LocalInferenceRun { output, profile })
    }

    pub fn infer_to_file(
        &self,
        request: &A::Request,
        options: &LocalInferenceOptions,
        path: impl AsRef<Path>,
    ) -> Result<LocalInferenceRun<A::Output>, A::Error> {
        let mut run = self.infer(request, options)?;
        run.profile.record_result("wav_write", || {
            self.adapter.write_output(&run.output, path.as_ref())
        })?;
        Ok(run)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use burn::backend::Flex;

    use super::*;

    type TestBackend = Flex;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FakeOutput {
        text: String,
    }

    #[derive(Debug)]
    struct FakeAdapter;

    impl LocalModelAdapter<TestBackend> for FakeAdapter {
        type Request = String;
        type Output = FakeOutput;
        type Error = std::io::Error;
        type LoadReport = &'static str;

        fn load(
            _model_dir: &Path,
            _device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> Result<Self, Self::Error> {
            Ok(Self)
        }

        fn load_report(&self) -> Self::LoadReport {
            "loaded"
        }

        fn infer(
            &self,
            request: &Self::Request,
            _options: &LocalInferenceOptions,
            profile: &mut LocalInferenceProfile,
        ) -> Result<Self::Output, Self::Error> {
            let output = profile.record("fake_stage", || FakeOutput {
                text: request.clone(),
            });
            Ok(output)
        }

        fn write_output(&self, _output: &Self::Output, _path: &Path) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn core_runs_model_agnostic_adapter_and_collects_profile() {
        let device = Default::default();
        let core = LocalInferenceCore::<TestBackend, FakeAdapter>::load(".", &device).unwrap();
        let run = core
            .infer(&"hello".to_string(), &LocalInferenceOptions::default())
            .unwrap();

        assert_eq!(core.load_report(), "loaded");
        assert_eq!(
            run.output,
            FakeOutput {
                text: "hello".to_string()
            }
        );
        assert_eq!(run.profile.stages.len(), 1);
        assert_eq!(run.profile.stages[0].name, "fake_stage");
    }
}
