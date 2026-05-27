"""Probe eager vs manual attention variants inside talker decode."""

import argparse
import json
from pathlib import Path
import types

import torch
from generate_reference_v9_code_predictor import load_model
from probe_v9_code_predictor_attention_kernel import (
    manual_attention,
    repeat_kv,
    tensor_summary,
)


def install_talker_kernel_probe(talker, target_step, captured):
    state = {"generation_step": None}
    original_forward = talker.forward
    attn_cls = type(talker.model.layers[0].self_attn)
    attn_globals = attn_cls.forward.__globals__
    apply_mrope = attn_globals["apply_multimodal_rotary_pos_emb"]

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
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ):
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            state["generation_step"] = -1
        else:
            state["generation_step"] = int(generation_step)
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
            past_hidden=past_hidden,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            generation_step=generation_step,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_p=subtalker_top_p,
            subtalker_top_k=subtalker_top_k,
            subtalker_temperature=subtalker_temperature,
            **kwargs,
        )
        state["generation_step"] = None
        return result

    talker.forward = types.MethodType(forward_with_state, talker)

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
            query_states, key_states = apply_mrope(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_scaling["mrope_section"],
                self.rope_scaling["interleaved"],
            )

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
            eager_output = self.o_proj(eager_pre_o.reshape(*input_shape, -1).contiguous())
            if state["generation_step"] == target_step:
                captured[str(layer_idx)] = {
                    "query_dtype": str(query_states.dtype),
                    "key_len": int(key_states.shape[-2]),
                    "query_len": int(query_states.shape[-2]),
                    "has_attention_mask": attention_mask is not None,
                    "is_causal": bool(is_causal),
                    "pre_o": {
                        mode: tensor_summary(eager_pre_o, value)
                        for mode, value in manual.items()
                    },
                    "post_o": {
                        mode: tensor_summary(
                            eager_output,
                            self.o_proj(value.reshape(*input_shape, -1).contiguous()),
                        )
                        for mode, value in manual.items()
                    },
                }
            return eager_output, None

        return attention_forward

    for layer_idx, layer in enumerate(talker.model.layers):
        layer.self_attn.forward = types.MethodType(
            make_attention_forward(layer_idx), layer.self_attn
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()

    wrapper = load_model(Path(args.model_dir))
    wrapper.model.eval()
    captured = {}
    install_talker_kernel_probe(wrapper.model.talker, args.step, captured)
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
    output = {
        "step": args.step,
        "base_token_ids": codes_list[0][:, 0].detach().cpu().to(torch.int64).tolist(),
        "talker_hidden_shape": list(hidden_states_list[0].shape),
        "layers": captured,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
