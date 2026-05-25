import torch
import json
import argparse
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

def cache_seq_len(past_key_values):
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    first_layer = past_key_values[0]
    key = first_layer[0] if isinstance(first_layer, tuple) else first_layer.keys
    return int(key.shape[-2])


def make_additive_causal_mask(attention_mask, dtype):
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=attention_mask.device, dtype=dtype)
    upper = torch.triu(
        torch.ones((seq_len, seq_len), device=attention_mask.device, dtype=torch.bool),
        diagonal=1,
    )
    mask = mask.masked_fill(upper.view(1, 1, seq_len, seq_len), float("-inf"))
    padding = attention_mask[:, None, None, :] == 0
    return mask.masked_fill(padding, float("-inf"))


def generate_reference(model_dir, output_json, input_text="Hello", max_new_tokens=4):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    # 1. Load Model
    config = Qwen3TTSTalkerConfig.from_pretrained(model_dir)
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(model_dir, dtype="auto").eval()
    model_dtype = next(model.parameters()).dtype
    
    # 2. Prepare dummy but deterministic input
    torch.manual_seed(60)
    batch_size = 1
    seq_len = 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size).to(model_dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(3, batch_size, 1)
    decode_inputs_embeds = torch.randn(batch_size, 1, config.hidden_size).to(model_dtype)
    decode_position_ids = torch.full((3, batch_size, 1), seq_len, dtype=torch.long)
    decode_attention_mask = torch.ones((batch_size, seq_len + 1), dtype=torch.long)
    
    # 3. Inference with hooks to catch layer outputs
    results = {"prefill": {}, "decode": {}}
    active_phase = {"name": "prefill"}

    def tensor_stats(output):
        flattened = output.flatten()
        return {
            "shape": list(output.shape),
            "first_5": flattened[:5].tolist(),
            "last_5": flattened[-5:].tolist(),
            "values": flattened.tolist(),
        }

    def collect_prefill_outputs():
        activations = {}
        hidden_states = inputs_embeds
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=inputs_embeds.device)
        causal_mask = make_additive_causal_mask(attention_mask, dtype=hidden_states.dtype)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            activations[f"layers.{layer_idx}.input_norm.output"] = tensor_stats(normalized)
            attn_output, _ = layer.self_attn(
                hidden_states=normalized,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=None,
                cache_position=None,
            )
            hidden_states = residual + attn_output
            activations[f"layers.{layer_idx}.attn.output"] = tensor_stats(attn_output)
            activations[f"layers.{layer_idx}.attn_residual.output"] = tensor_stats(hidden_states)

            residual = hidden_states
            mlp_input = layer.post_attention_layernorm(hidden_states)
            activations[f"layers.{layer_idx}.post_attention_norm.output"] = tensor_stats(mlp_input)
            mlp_output = layer.mlp(mlp_input)
            hidden_states = residual + mlp_output
            activations[f"layers.{layer_idx}.mlp.output"] = tensor_stats(mlp_output)
            activations[f"layers.{layer_idx}.hidden.output"] = tensor_stats(hidden_states)

        hidden_states = model.model.norm(hidden_states)
        activations["final_norm"] = tensor_stats(hidden_states)
        logits = model.codec_head(hidden_states)
        return activations, hidden_states, logits

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            results.setdefault(active_phase["name"], {})[name] = tensor_stats(output)
        return fn

    # Register hooks for decoder layer outputs so Rust can locate the first drift.
    for layer_idx, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_fn(f"layers.{layer_idx}.hidden.output"))
    model.model.norm.register_forward_hook(hook_fn("final_norm"))

    generation_steps = []
    generated_token_ids = []

    with torch.no_grad():
        active_phase["name"] = "prefill"
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = model.codec_head(outputs.last_hidden_state)
        manual_prefill_activations, manual_hidden_states, manual_logits = collect_prefill_outputs()
        prefill_selected_token_ids = logits[:, -1, :].argmax(dim=-1)
        generated_token_ids.append(prefill_selected_token_ids)

        active_phase["name"] = "decode"
        decode_outputs = model.model(
            inputs_embeds=decode_inputs_embeds,
            position_ids=decode_position_ids,
            attention_mask=decode_attention_mask,
            past_key_values=outputs.past_key_values,
            cache_position=torch.arange(seq_len, seq_len + 1),
            use_cache=True,
        )
        decode_logits = model.codec_head(decode_outputs.last_hidden_state)

        active_phase["name"] = "generation_prefill"
        generation_prefill_outputs = model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
        )
        generation_cache = generation_prefill_outputs.past_key_values
        selected_token_ids = prefill_selected_token_ids
        for step_idx in range(1, max_new_tokens):
            cache_len_before = cache_seq_len(generation_cache)
            step_inputs_embeds = model.get_input_embeddings()(selected_token_ids.unsqueeze(1))
            step_position_ids = torch.full(
                (3, batch_size, 1),
                cache_len_before,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            step_attention_mask = torch.ones(
                (batch_size, cache_len_before + 1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            active_phase["name"] = f"generation_step_{step_idx}"
            step_outputs = model.model(
                inputs_embeds=step_inputs_embeds,
                position_ids=step_position_ids,
                attention_mask=step_attention_mask,
                past_key_values=generation_cache,
                cache_position=torch.arange(
                    cache_len_before,
                    cache_len_before + 1,
                    device=inputs_embeds.device,
                ),
                use_cache=True,
            )
            step_logits = model.codec_head(step_outputs.last_hidden_state)
            selected_token_ids = step_logits[:, -1, :].argmax(dim=-1)
            generated_token_ids.append(selected_token_ids)
            generation_cache = step_outputs.past_key_values
            generation_steps.append(
                {
                    "token_id": selected_token_ids.tolist(),
                    "logits": tensor_stats(step_logits),
                    "cache_len_before": cache_len_before,
                    "cache_len_after": cache_seq_len(generation_cache),
                    "hidden": {
                        "output": tensor_stats(step_outputs.last_hidden_state),
                    },
                }
            )
        
    # 4. Save to JSON
    reference_data = {
        "input": {
            "inputs_embeds": inputs_embeds.tolist(),
            "position_ids": position_ids.tolist(),
        },
        "decode_input": {
            "inputs_embeds": decode_inputs_embeds.tolist(),
            "position_ids": decode_position_ids.tolist(),
            "attention_mask": decode_attention_mask.tolist(),
        },
        "expected": {
            "logits": tensor_stats(manual_logits),
            "layer_0_output": manual_prefill_activations["layers.0.hidden.output"],
            **manual_prefill_activations
        },
        "decode_expected": {
            "logits": tensor_stats(decode_logits),
            "last_hidden_state": tensor_stats(decode_outputs.last_hidden_state),
            "cache_len_before": seq_len,
            "cache_len_after": seq_len + 1,
            "layer_0_output": results["decode"]["layers.0.hidden.output"],
            **results["decode"],
        },
        "generation_input": {
            "max_new_tokens": max_new_tokens,
        },
        "generation_expected": {
            "generated_token_ids": torch.stack(generated_token_ids, dim=1).tolist(),
            "prefill_selected_token_id": prefill_selected_token_ids.tolist(),
            "steps": generation_steps,
        },
    }
    
    with open(output_json, 'w') as f:
        json.dump(reference_data, f, indent=2)
    print(f"Reference data saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="reference.json")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    generate_reference(args.model_dir, args.output, max_new_tokens=args.max_new_tokens)
