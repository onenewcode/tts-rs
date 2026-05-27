"""Generate V9 talker decode-step activation reference data."""

import argparse
import json
from pathlib import Path
import types

import torch
from qwen_tts import Qwen3TTSModel


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


def install_decode_capture(talker, target_steps, max_layers=None):
    captured = {step: {} for step in target_steps}
    state = {"generation_step": None}
    original_talker_forward = talker.forward

    def talker_forward_with_state(
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
        result = original_talker_forward(
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
        step = state["generation_step"]
        if step in captured:
            if inputs_embeds is None:
                codec_ids = result.hidden_states[1]
                codec_hiddens = torch.cat(
                    [self.get_input_embeddings()(codec_ids[..., 0:1])]
                    + [
                        self.code_predictor.get_input_embeddings()[i](
                            codec_ids[..., i + 1 : i + 2]
                        )
                        for i in range(self.config.num_code_groups - 1)
                    ],
                    dim=1,
                )
                captured_inputs = codec_hiddens.sum(1, keepdim=True)
                if generation_step < trailing_text_hidden.shape[1]:
                    captured_inputs = captured_inputs + trailing_text_hidden[
                        :, generation_step
                    ].unsqueeze(1)
                else:
                    captured_inputs = captured_inputs + tts_pad_embed
            else:
                captured_inputs = inputs_embeds
            captured[step]["decode.inputs_embeds"] = captured_inputs.detach().cpu()
            captured[step]["model.norm.output"] = result.past_hidden.detach().cpu()
            captured[step]["codec_head.logits"] = result.logits.detach().cpu()
        state["generation_step"] = None
        return result


    talker.forward = types.MethodType(talker_forward_with_state, talker)

    for layer_idx, layer in enumerate(talker.model.layers):
        if max_layers is not None and layer_idx >= max_layers:
            continue
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
                step = state["generation_step"]
                should_capture = step in captured
                residual = hidden_states
                input_norm = self.input_layernorm(hidden_states)
                attn_output, self_attn_weights = self.self_attn(
                    hidden_states=input_norm,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
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
                    captured[step][f"{prefix}.input_norm.output"] = input_norm.detach().cpu()
                    captured[step][f"{prefix}.attn.output"] = attn_output.detach().cpu()
                    captured[step][f"{prefix}.attn_residual.output"] = attn_residual.detach().cpu()
                    captured[step][f"{prefix}.post_attention_norm.output"] = post_attention_norm.detach().cpu()
                    captured[step][f"{prefix}.mlp.gate"] = gate.detach().cpu()
                    captured[step][f"{prefix}.mlp.up"] = up.detach().cpu()
                    captured[step][f"{prefix}.mlp.activated_gate"] = activated_gate.detach().cpu()
                    captured[step][f"{prefix}.mlp.product"] = product.detach().cpu()
                    captured[step][f"{prefix}.mlp.output"] = mlp_output.detach().cpu()
                    captured[step][f"{prefix}.hidden.output"] = hidden.detach().cpu()

                outputs = (hidden,)
                if output_attentions:
                    outputs += (self_attn_weights,)
                return outputs

            return layer_forward_with_capture

        layer.forward = types.MethodType(make_layer_forward(layer_idx, original_layer_forward), layer)

    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    parser.add_argument("--steps", default="0,1,2,3,4")
    parser.add_argument("--max-layers", type=int, default=None)
    args = parser.parse_args()

    wrapper = load_model(Path(args.model_dir))
    wrapper.model.eval()
    target_steps = {int(item) for item in args.steps.split(",") if item}
    captured = install_decode_capture(wrapper.model.talker, target_steps, args.max_layers)
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": args.text,
                "language": args.language,
                "speaker": args.speaker,
                "max_new_tokens": args.max_new_tokens,
                "base_token_ids": codes_list[0][:, 0].detach().cpu().to(torch.int64).tolist(),
                "codec_groups": codes_list[0].detach().cpu().to(torch.int64).tolist(),
                "talker_hidden": tensor_stats(hidden_states_list[0].detach().cpu()),
                "steps": [
                    {
                        "generation_step": step,
                        "activations": {
                            name: tensor_stats(tensor)
                            for name, tensor in captured[step].items()
                        },
                    }
                    for step in sorted(captured)
                    if captured[step]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
