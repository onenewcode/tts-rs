"""Probe eager vs manual attention variants inside code predictor."""

import argparse
import json
from pathlib import Path
import types

import torch
from generate_reference_v9_code_predictor import load_model


def tensor_summary(left, right):
    diff = (left.detach().cpu().float() - right.detach().cpu().float()).abs().flatten()
    max_abs, max_idx = torch.max(diff, dim=0)
    return {
        "max_abs": float(max_abs.item()),
        "max_idx": int(max_idx.item()),
        "exceed_1e_3": int((diff > 1e-3).sum().item()),
        "exceed_5e_2": int((diff > 5e-2).sum().item()),
    }


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def manual_attention(module, query, key, value, attention_mask, mode, is_causal=False):
    if not use_gqa_in_manual_attention(attention_mask, key):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if mode == "eager":
        scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        elif is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value)
    elif mode == "rust_current":
        scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        elif is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.nn.functional.softmax(scores, dim=-1).to(query.dtype)
        output = torch.matmul(weights.float(), value.float()).to(query.dtype)
    elif mode == "f32_all_then_cast":
        scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        elif is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(weights, value.float()).to(query.dtype)
    elif mode == "bf16_scores_f32_value":
        scores = torch.matmul(query, key.transpose(2, 3)).float() * module.scaling
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        elif is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.nn.functional.softmax(scores, dim=-1).to(query.dtype)
        output = torch.matmul(weights.float(), value.float()).to(query.dtype)
    else:
        raise ValueError(mode)

    return output.transpose(1, 2).contiguous()


def use_gqa_in_manual_attention(attention_mask, key):
    # Match transformers' eager CPU behavior for this model: pre-repeat K/V.
    # The generated masks are 4D.
    return False


def install_kernel_probe(code_predictor, target_head, target_layer, captured):
    state = {"head_idx": None}
    original_forward = code_predictor.forward
    attn_cls = type(code_predictor.model.layers[target_layer].self_attn)
    attn_globals = attn_cls.forward.__globals__
    apply_rotary_pos_emb = attn_globals["apply_rotary_pos_emb"]

    def normalize_generation_step(value):
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().flatten()[0].item())
        return int(value)

    def forward_with_state(
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
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            generation_steps=generation_steps,
            **kwargs,
        )
        state["head_idx"] = None
        return result

    code_predictor.forward = types.MethodType(forward_with_state, code_predictor)

    def make_attention_forward(layer_idx):
        def attention_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            is_causal = (
                query_states.shape[2] > 1
                and attention_mask is None
                and getattr(self, "is_causal", True)
            )
            eager_pre_o = manual_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                "eager",
                is_causal=is_causal,
            )

            eager_output = self.o_proj(eager_pre_o.reshape(*input_shape, -1).contiguous())
            if state["head_idx"] == target_head and layer_idx == target_layer:
                manual = {
                    mode: manual_attention(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        mode,
                        is_causal=is_causal,
                    )
                    for mode in [
                        "eager",
                        "rust_current",
                        "f32_all_then_cast",
                        "bf16_scores_f32_value",
                    ]
                }
                captured["query_dtype"] = str(query_states.dtype)
                captured["key_len"] = int(key_states.shape[-2])
                captured["query_len"] = int(query_states.shape[-2])
                captured["has_attention_mask"] = attention_mask is not None
                captured["is_causal"] = bool(is_causal)
                captured["pre_o"] = {
                    mode: tensor_summary(eager_pre_o, value)
                    for mode, value in manual.items()
                }
                captured["post_o"] = {
                    mode: tensor_summary(
                        eager_output,
                        self.o_proj(value.reshape(*input_shape, -1).contiguous()),
                    )
                    for mode, value in manual.items()
                }

            return eager_output, None

        return attention_forward

    layer = code_predictor.model.layers[target_layer]
    layer.self_attn.forward = types.MethodType(
        make_attention_forward(target_layer), layer.self_attn
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    args = parser.parse_args()

    wrapper = load_model(Path(args.model_dir))
    wrapper.model.eval()
    captured = {}
    install_kernel_probe(
        wrapper.model.talker.code_predictor, args.head, args.layer, captured
    )

    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(args.text)])
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
    base_id = codes[args.step : args.step + 1, 0:1].to(torch.long)
    talker_hidden = hidden_states[args.step : args.step + 1].unsqueeze(1)
    base_embed = wrapper.model.talker.get_input_embeddings()(base_id)
    predictor = wrapper.model.talker.code_predictor.generate(
        inputs_embeds=torch.cat((talker_hidden, base_embed), dim=1),
        max_new_tokens=wrapper.model.talker.config.num_code_groups - 1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    output = {
        "step": args.step,
        "head": args.head,
        "layer": args.layer,
        "base_token_id": int(base_id.item()),
        "codec_groups": (
            torch.cat((base_id, predictor.sequences), dim=-1)
            .detach()
            .cpu()
            .to(torch.int64)
            .squeeze(0)
            .tolist()
        ),
        "captured": captured,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
