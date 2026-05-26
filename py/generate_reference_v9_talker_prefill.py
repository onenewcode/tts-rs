"""Generate V9 talker prefill activation reference data."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb


def tensor_stats(tensor):
    flat = tensor.detach().cpu().float().flatten()
    return {
        "shape": list(tensor.shape),
        "values": flat.tolist(),
        "num_elements": flat.numel(),
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
    }


def load_model(model_dir):
    try:
        return Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", dtype=torch.bfloat16
        )
    except TypeError:
        return Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", torch_dtype=torch.bfloat16
        )


def capture_frontend_inputs(wrapper, input_ids, language, speaker):
    captured = {}

    def capture_generate(**kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().cpu()
        captured["attention_mask"] = kwargs["attention_mask"].detach().cpu()
        raise RuntimeError("__captured_prefill__")

    original_generate = wrapper.model.talker.generate
    wrapper.model.talker.generate = capture_generate
    try:
        wrapper.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            do_sample=False,
            max_new_tokens=2,
            repetition_penalty=1.0,
        )
    except RuntimeError as exc:
        if str(exc) != "__captured_prefill__":
            raise
    finally:
        wrapper.model.talker.generate = original_generate
    return captured


def flatten_heads(x):
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, seq_len, heads * head_dim)


@torch.no_grad()
def collect_prefill_activations(talker, inputs_embeds, attention_mask, max_layers=None):
    activations = {}
    hidden_states = inputs_embeds
    position_ids = attention_mask.to(torch.int64).cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    position_embeddings = talker.model.rotary_emb(hidden_states, position_ids)

    batch_size, seq_len = attention_mask.shape
    causal_mask = torch.zeros(
        (batch_size, 1, seq_len, seq_len),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    causal_mask = causal_mask.masked_fill(
        torch.triu(
            torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len),
        float("-inf"),
    )
    causal_mask = causal_mask.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))

    layer_count = len(talker.model.layers) if max_layers is None else min(max_layers, len(talker.model.layers))
    for layer_idx, layer in enumerate(talker.model.layers[:layer_count]):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        activations[f"layers.{layer_idx}.input_norm.output"] = normalized

        input_shape = normalized.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        q_proj = layer.self_attn.q_proj(normalized)
        k_proj = layer.self_attn.k_proj(normalized)
        v_proj = layer.self_attn.v_proj(normalized)
        q_norm = layer.self_attn.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)
        k_norm = layer.self_attn.k_norm(k_proj.view(hidden_shape)).transpose(1, 2)
        q_rot, k_rot = apply_multimodal_rotary_pos_emb(
            q_norm,
            k_norm,
            position_embeddings[0],
            position_embeddings[1],
            layer.self_attn.rope_scaling["mrope_section"],
            layer.self_attn.rope_scaling["interleaved"],
        )
        activations[f"layers.{layer_idx}.q_proj.output"] = q_proj
        activations[f"layers.{layer_idx}.k_proj.output"] = k_proj
        activations[f"layers.{layer_idx}.v_proj.output"] = v_proj
        activations[f"layers.{layer_idx}.q_norm.output"] = flatten_heads(q_norm)
        activations[f"layers.{layer_idx}.k_norm.output"] = flatten_heads(k_norm)
        activations[f"layers.{layer_idx}.q_rot.output"] = flatten_heads(q_rot)
        activations[f"layers.{layer_idx}.k_rot.output"] = flatten_heads(k_rot)

        attn_output, _ = layer.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=None,
        )
        hidden_states = residual + attn_output
        activations[f"layers.{layer_idx}.attn.output"] = attn_output
        activations[f"layers.{layer_idx}.attn_residual.output"] = hidden_states

        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)
        activations[f"layers.{layer_idx}.post_attention_norm.output"] = mlp_input
        gate = layer.mlp.gate_proj(mlp_input)
        up = layer.mlp.up_proj(mlp_input)
        activated_gate = layer.mlp.act_fn(gate)
        product = activated_gate * up
        mlp_output = layer.mlp.down_proj(product)
        hidden_states = residual + mlp_output
        activations[f"layers.{layer_idx}.mlp.gate"] = gate
        activations[f"layers.{layer_idx}.mlp.up"] = up
        activations[f"layers.{layer_idx}.mlp.activated_gate"] = activated_gate
        activations[f"layers.{layer_idx}.mlp.product"] = product
        activations[f"layers.{layer_idx}.mlp.output"] = mlp_output
        activations[f"layers.{layer_idx}.hidden.output"] = hidden_states

    if layer_count == len(talker.model.layers):
        hidden_states = talker.model.norm(hidden_states)
        activations["model.norm.output"] = hidden_states
        logits = talker.codec_head(hidden_states)
        activations["codec_head.logits"] = logits

    return position_ids, activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-layers", type=int, default=4)
    args = parser.parse_args()

    wrapper = load_model(Path(args.model_dir))
    wrapper.model.eval()
    prompt = wrapper._build_assistant_text(args.text)
    input_ids = wrapper._tokenize_texts([prompt])
    captured = capture_frontend_inputs(wrapper, input_ids, args.language, args.speaker)
    inputs_embeds = captured["inputs_embeds"].to(next(wrapper.model.parameters()).device)
    attention_mask = captured["attention_mask"].to(next(wrapper.model.parameters()).device)
    position_ids, activations = collect_prefill_activations(
        wrapper.model.talker,
        inputs_embeds,
        attention_mask,
        args.max_layers,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": args.text,
                "language": args.language,
                "speaker": args.speaker,
                "attention_mask": attention_mask.detach().cpu().to(torch.int64).tolist(),
                "position_ids": position_ids.detach().cpu().to(torch.int64).tolist(),
                "inputs_embeds": tensor_stats(inputs_embeds),
                "max_layers": args.max_layers,
                "activations": {
                    name: tensor_stats(tensor) for name, tensor in activations.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
