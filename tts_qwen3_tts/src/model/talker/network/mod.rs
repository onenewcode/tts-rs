pub(crate) mod attention;
pub(crate) mod kv;
pub(crate) mod layer;
mod mask;
pub(crate) mod mlp;
mod model;
mod predictor;
pub(crate) mod rope;

pub(crate) use self::mask::build_attention_mask;
pub use self::model::{Qwen3TtsTalker, Qwen3TtsTalkerModel, Qwen3TtsTalkerModelBundle};
pub use self::predictor::{Qwen3TtsTalkerCodePredictor, Qwen3TtsTalkerCodePredictorModel};
