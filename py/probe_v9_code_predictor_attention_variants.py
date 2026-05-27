"""Compare code-predictor generation under eager/manual attention variants."""

import argparse
import json
from pathlib import Path
import types

import torch

from generate_reference_v9_code_predictor import load_model
from probe_v9_code_predictor_attention_kernel import manual_attention, repeat_kv


def install_manual_attention(code_predictor, mode):
    attn_cls = type(code_predictor.model.layers[0].self_attn)
    attn_globals = attn_cls.forward.__globals__
    apply_rotary_pos_emb = attn_globals["apply_rotary_pos_emb"]

    def make_attention_forward():
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

            if attention_mask is not None and attention_mask.ndim == 4:
                attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            is_causal = (
                query_states.shape[2] > 1
                and attention_mask is None
                and getattr(self, "is_causal", True)
            )
            output = manual_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                mode,
                is_causal=is_causal,
            )
            output = self.o_proj(output.reshape(*input_shape, -1).contiguous())
            return output, None

        return attention_forward

    for layer in code_predictor.model.layers:
        layer.self_attn.forward = types.MethodType(make_attention_forward(), layer.self_attn)


@torch.no_grad()
def predictor_groups(wrapper, codes, hidden_states, step_idx):
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
    return (
        torch.cat((base_id, predictor.sequences), dim=-1)
        .detach()
        .cpu()
        .to(torch.int64)
        .squeeze(0)
        .tolist()
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    default_wrapper = load_model(model_dir)
    default_wrapper.model.eval()
    input_ids = default_wrapper._tokenize_texts(
        [default_wrapper._build_assistant_text(args.text)]
    )
    codes_list, hidden_states_list = default_wrapper.model.generate(
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
    expected = predictor_groups(default_wrapper, codes, hidden_states, args.step)

    variants = {}
    for mode in ["eager", "rust_current", "f32_all_then_cast", "bf16_scores_f32_value"]:
        wrapper = load_model(model_dir)
        wrapper.model.eval()
        install_manual_attention(wrapper.model.talker.code_predictor, mode)
        variants[mode] = predictor_groups(wrapper, codes, hidden_states, args.step)

    output = {
        "step": args.step,
        "expected_eager_groups": expected,
        "variants": variants,
        "matches": {mode: groups == expected for mode, groups in variants.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
