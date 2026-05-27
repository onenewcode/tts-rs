"""Generate V9 code-predictor alignment reference data for selected steps."""

import argparse
import json
from pathlib import Path
import types

import torch
from qwen_tts import Qwen3TTSModel


def tensor_values(tensor):
    flat = tensor.detach().cpu().float().flatten()
    return {
        "shape": list(tensor.shape),
        "values": flat.tolist(),
        "num_elements": flat.numel(),
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
    }


def score_topk(scores, k=5):
    topk = []
    for score in scores:
        last_score = score.detach().cpu().float().reshape(-1, score.shape[-1])[-1]
        values, indices = torch.topk(last_score, k=min(k, last_score.shape[-1]), dim=-1)
        topk.append(
            {
                "ids": indices.to(torch.int64).tolist(),
                "values": values.tolist(),
            }
        )
    return topk


def score_tensors(scores):
    return [tensor_values(score.detach().cpu().float()) for score in scores]


def capture_cache_tensors(cache):
    if cache is None:
        return {}

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is None or value_cache is None:
        layers = getattr(cache, "layers", None)
        if layers is None:
            return {}
        key_cache = [getattr(layer, "keys", None) for layer in layers]
        value_cache = [getattr(layer, "values", None) for layer in layers]

    tensors = {}
    for layer_idx, (key, value) in enumerate(zip(key_cache, value_cache)):
        if key is not None:
            tensors[f"layers.{layer_idx}.cache.key"] = key.detach().cpu()
        if value is not None:
            tensors[f"layers.{layer_idx}.cache.value"] = value.detach().cpu()
    return tensors


def cache_layer_tensor(cache, layer_idx, value_name):
    if cache is None:
        return None

    cache_list = getattr(cache, f"{value_name}_cache", None)
    if cache_list is not None:
        return cache_list[layer_idx]

    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        plural_name = "keys" if value_name == "key" else "values"
        return getattr(layer, plural_name, None)

    try:
        pair = cache[layer_idx]
    except (TypeError, IndexError):
        return None
    return pair[0 if value_name == "key" else 1]


def repeat_kv_for_capture(hidden_states, n_rep):
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def load_model(model_dir):
    try:
        wrapper = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", dtype=torch.bfloat16
        )
    except TypeError:
        wrapper = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", torch_dtype=torch.bfloat16
        )
    force_eager_attention(wrapper)
    return wrapper


def force_eager_attention(wrapper):
    """Keep Python references on eager attention, not PyTorch SDPA kernels."""
    model = getattr(wrapper, "model", None)
    talker = getattr(model, "talker", None)
    code_predictor = getattr(talker, "code_predictor", None)
    candidates = [
        getattr(wrapper, "config", None),
        getattr(model, "config", None),
        getattr(talker, "config", None),
        getattr(getattr(talker, "model", None), "config", None),
        getattr(code_predictor, "config", None),
        getattr(getattr(code_predictor, "model", None), "config", None),
    ]
    for config in candidates:
        if config is not None:
            config._attn_implementation = "eager"


def install_code_predictor_capture(code_predictor):
    captured = {}
    state = {"head_idx": None}
    original_forward = code_predictor.forward
    rotary_fn = type(code_predictor.model.layers[0].self_attn).forward.__globals__[
        "apply_rotary_pos_emb"
    ]

    def normalize_generation_step(value):
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().flatten()[0].item())
        return int(value)

    def forward_with_capture(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ):
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            state["head_idx"] = inputs_embeds.shape[1] - 2
        else:
            state["head_idx"] = normalize_generation_step(generation_steps)
        result = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            generation_steps=generation_steps,
            **kwargs,
        )
        head_idx = state["head_idx"]
        captured.setdefault(head_idx, {})["model.norm.output"] = (
            result.hidden_states[-1].detach().cpu()
            if result.hidden_states is not None
            else result.logits.detach().cpu()[:, :, :0]
        )
        captured.setdefault(head_idx, {})["lm_head.logits"] = result.logits.detach().cpu()
        captured.setdefault(head_idx, {}).update(capture_cache_tensors(result.past_key_values))
        state["head_idx"] = None
        return result

    code_predictor.forward = types.MethodType(forward_with_capture, code_predictor)

    for layer_idx, layer in enumerate(code_predictor.model.layers):
        original_layer_forward = layer.forward

        def make_layer_forward(layer_idx, original_layer_forward):
            def layer_forward_with_capture(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                head_idx = state["head_idx"]
                should_capture = head_idx is not None
                residual = hidden_states
                input_norm = self.input_layernorm(hidden_states)
                attn_output, self_attn_weights = self.self_attn(
                    hidden_states=input_norm,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                attn_residual = residual + attn_output

                residual = attn_residual
                post_attention_norm = self.post_attention_layernorm(attn_residual)
                gate = self.mlp.gate_proj(post_attention_norm)
                up = self.mlp.up_proj(post_attention_norm)
                activated_gate = self.mlp.act_fn(gate)
                product = activated_gate * up
                mlp_output = self.mlp.down_proj(product)
                hidden = residual + mlp_output

                if should_capture:
                    prefix = f"layers.{layer_idx}"
                    bucket = captured.setdefault(head_idx, {})
                    input_shape = input_norm.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
                    q_proj = self.self_attn.q_proj(input_norm)
                    k_proj = self.self_attn.k_proj(input_norm)
                    v_proj = self.self_attn.v_proj(input_norm)
                    q_norm = self.self_attn.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)
                    k_norm = self.self_attn.k_norm(k_proj.view(hidden_shape)).transpose(1, 2)
                    cos, sin = position_embeddings
                    q_rot, k_rot = rotary_fn(q_norm, k_norm, cos, sin)
                    key_cache = cache_layer_tensor(past_key_values, layer_idx, "key")
                    if key_cache is not None:
                        key_states = repeat_kv_for_capture(
                            key_cache, self.self_attn.num_key_value_groups
                        )
                        scores = (
                            torch.matmul(q_rot, key_states.transpose(2, 3))
                            * self.self_attn.scaling
                        )
                        if attention_mask is not None:
                            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                            scores = scores + causal_mask
                        manual_weights = torch.nn.functional.softmax(
                            scores, dim=-1, dtype=torch.float32
                        ).to(q_rot.dtype)
                        bucket[f"{prefix}.attn.scores"] = scores.detach().cpu()
                        bucket[f"{prefix}.attn.manual_weights"] = manual_weights.detach().cpu()
                    bucket[f"{prefix}.input_norm.output"] = input_norm.detach().cpu()
                    bucket[f"{prefix}.attn.output"] = attn_output.detach().cpu()
                    if self_attn_weights is not None:
                        bucket[f"{prefix}.attn.weights"] = self_attn_weights.detach().cpu()
                    bucket[f"{prefix}.attn_residual.output"] = attn_residual.detach().cpu()
                    bucket[f"{prefix}.post_attention_norm.output"] = post_attention_norm.detach().cpu()
                    bucket[f"{prefix}.q_proj.output"] = q_proj.detach().cpu()
                    bucket[f"{prefix}.k_proj.output"] = k_proj.detach().cpu()
                    bucket[f"{prefix}.v_proj.output"] = v_proj.detach().cpu()
                    bucket[f"{prefix}.q_norm.output"] = (
                        q_norm.transpose(1, 2).reshape(*input_shape, -1).detach().cpu()
                    )
                    bucket[f"{prefix}.k_norm.output"] = (
                        k_norm.transpose(1, 2).reshape(*input_shape, -1).detach().cpu()
                    )
                    bucket[f"{prefix}.q_rot.output"] = (
                        q_rot.transpose(1, 2).reshape(*input_shape, -1).detach().cpu()
                    )
                    bucket[f"{prefix}.k_rot.output"] = (
                        k_rot.transpose(1, 2).reshape(*input_shape, -1).detach().cpu()
                    )
                    bucket[f"{prefix}.mlp.gate"] = gate.detach().cpu()
                    bucket[f"{prefix}.mlp.up"] = up.detach().cpu()
                    bucket[f"{prefix}.mlp.activated_gate"] = activated_gate.detach().cpu()
                    bucket[f"{prefix}.mlp.product"] = product.detach().cpu()
                    bucket[f"{prefix}.mlp.output"] = mlp_output.detach().cpu()
                    bucket[f"{prefix}.hidden.output"] = hidden.detach().cpu()

                outputs = (hidden,)
                if output_attentions:
                    outputs += (self_attn_weights,)
                return outputs

            return layer_forward_with_capture

        layer.forward = types.MethodType(make_layer_forward(layer_idx, original_layer_forward), layer)

    return captured


@torch.no_grad()
def predictor_reference(wrapper, codes, hidden_states, step_idx, captured):
    captured.clear()
    base_id = codes[step_idx : step_idx + 1, 0:1].to(torch.long)
    talker_hidden = hidden_states[step_idx : step_idx + 1].unsqueeze(1)
    base_embed = wrapper.model.talker.get_input_embeddings()(base_id)
    predictor = wrapper.model.talker.code_predictor.generate(
        inputs_embeds=torch.cat((talker_hidden, base_embed), dim=1),
        max_new_tokens=wrapper.model.talker.config.num_code_groups - 1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    groups = (
        torch.cat((base_id, predictor.sequences), dim=-1)
        .detach()
        .cpu()
        .to(torch.int64)
        .squeeze(0)
        .tolist()
    )
    generated_activations = {
        head_idx: dict(head_activations)
        for head_idx, head_activations in captured.items()
    }
    teacher_forced_scores = teacher_forced_predictor_scores(
        wrapper, groups, hidden_states[step_idx : step_idx + 1]
    )
    return {
        "step_idx": step_idx,
        "base_token_id": int(base_id.item()),
        "expected_codec_groups": groups,
        "talker_hidden": tensor_values(talker_hidden.squeeze(1)),
        "topk": score_topk(predictor.scores),
        "scores": score_tensors(predictor.scores),
        "teacher_forced_scores": score_tensors(teacher_forced_scores),
        "teacher_forced_topk": score_topk(teacher_forced_scores),
        "activations": {
            str(head_idx): {
                name: tensor_values(tensor)
                for name, tensor in sorted(head_activations.items())
            }
            for head_idx, head_activations in sorted(generated_activations.items())
        },
    }


@torch.no_grad()
def teacher_forced_predictor_scores(wrapper, codec_groups, talker_hidden):
    talker = wrapper.model.talker
    code_predictor = talker.code_predictor
    groups = torch.tensor(codec_groups, dtype=torch.long, device=talker_hidden.device).unsqueeze(0)
    base_embed = talker.get_input_embeddings()(groups[:, 0:1])
    output = code_predictor(
        inputs_embeds=torch.cat((talker_hidden.unsqueeze(1), base_embed), dim=1),
        use_cache=True,
    )
    scores = [output.logits.detach().cpu()]
    cache = output.past_key_values
    for head_idx in range(1, talker.config.num_code_groups - 1):
        cache_len = cache.get_seq_length() if hasattr(cache, "get_seq_length") else cache[0][0].shape[-2]
        output = code_predictor(
            input_ids=groups[:, head_idx : head_idx + 1],
            past_key_values=cache,
            cache_position=torch.arange(
                cache_len,
                cache_len + 1,
                device=talker_hidden.device,
            ),
            use_cache=True,
            generation_steps=head_idx,
        )
        scores.append(output.logits.detach().cpu())
        cache = output.past_key_values
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    parser.add_argument("--steps", default="0,1,2,3,4,5")
    parser.add_argument("--attention-implementation", choices=["eager"], default="eager")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    wrapper = load_model(model_dir)
    wrapper.model.eval()
    captured = install_code_predictor_capture(wrapper.model.talker.code_predictor)
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(args.text)])
    with torch.no_grad():
        codes_list, hidden_states_list = wrapper.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
            repetition_penalty=1.0,
        )

    codes = codes_list[0].detach().cpu().to(torch.int64)
    hidden_states = hidden_states_list[0]
    requested_steps = [int(item) for item in args.steps.split(",") if item]
    steps = [
        predictor_reference(wrapper, codes, hidden_states, step_idx, captured)
        for step_idx in requested_steps
        if 0 <= step_idx < codes.shape[0]
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": args.text,
                "language": args.language,
                "speaker": args.speaker,
                "max_new_tokens": args.max_new_tokens,
                "full_codec_groups": codes.tolist(),
                "steps": steps,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
