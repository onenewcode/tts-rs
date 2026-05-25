use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::DType;
use burn::tensor::Tensor;
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;

pub struct Qwen3TtsTextMlpOutput<B: Backend> {
    pub output: Tensor<B, 3>,
    pub gate: Tensor<B, 3>,
    pub up: Tensor<B, 3>,
    pub activated_gate: Tensor<B, 3>,
    pub product: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTextMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B> Qwen3TtsTextMlp<B>
where
    B: Backend,
{
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_activations(x).output
    }

    pub fn forward_with_activations(&self, x: Tensor<B, 3>) -> Qwen3TtsTextMlpOutput<B> {
        let dtype = x.dtype();
        let gate = native_linear_3d(&self.gate_proj, x.clone());
        let up = native_linear_3d(&self.up_proj, x);
        let activated_gate = silu(gate.clone().cast(DType::F32)).cast(dtype);
        let product = activated_gate.clone() * up.clone();
        let output = native_linear_3d(&self.down_proj, product.clone());
        Qwen3TtsTextMlpOutput {
            output,
            gate,
            up,
            activated_gate,
            product,
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3TtsTalkerResizeMlp<B: Backend> {
    pub linear_fc1: Linear<B>,
    pub linear_fc2: Linear<B>,
}

impl<B> Qwen3TtsTalkerResizeMlp<B>
where
    B: Backend,
{
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = native_linear_3d(&self.linear_fc1, x);
        let dtype = x.dtype();
        let x = silu(x.cast(DType::F32)).cast(dtype);
        native_linear_3d(&self.linear_fc2, x)
    }
}

fn native_linear_3d<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, _input_size] = x.dims();
    let output_size = linear.weight.dims()[1];
    let x_2d = x.reshape([batch_size * seq_len, _input_size]);
    let output_2d = linear.forward(x_2d);
    output_2d.reshape([batch_size, seq_len, output_size])
}
