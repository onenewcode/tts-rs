pub(crate) mod attention;
pub(crate) mod kv;
pub(crate) mod layer;
pub(crate) mod mlp;
mod model;
mod predictor;
pub(crate) mod rope;

pub use self::model::{Qwen3TtsTalker, Qwen3TtsTalkerModel, Qwen3TtsTalkerModelBundle};
pub use self::predictor::{Qwen3TtsTalkerCodePredictor, Qwen3TtsTalkerCodePredictorModel};
