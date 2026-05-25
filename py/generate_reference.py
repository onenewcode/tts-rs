import torch
import json
import argparse
import os
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

def generate_reference(model_dir, output_json, input_text="Hello"):
    # 1. Load Model
    config = Qwen3TTSTalkerConfig.from_pretrained(model_dir)
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(model_dir, dtype="auto").eval()
    model_dtype = next(model.parameters()).dtype
    
    # 2. Prepare dummy but deterministic input
    torch.manual_seed(42)
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

    def tensor_stats(output, include_values=False):
        stats = {
            "shape": list(output.shape),
            "sum": output.float().sum().item(),
            "mean": output.float().mean().item(),
            "first_5": output.flatten()[:5].tolist(),
        }
        if include_values:
            stats["values"] = output.flatten().tolist()
        return stats

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            results[active_phase["name"]][name] = tensor_stats(output)
        return fn

    # Register hooks for decoder layer outputs so Rust can locate the first drift.
    for layer_idx, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_fn(f"layers.{layer_idx}.hidden.output"))
    model.model.norm.register_forward_hook(hook_fn("final_norm"))

    with torch.no_grad():
        active_phase["name"] = "prefill"
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = model.codec_head(outputs.last_hidden_state)

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
            "logits": {
                "shape": list(logits.shape),
                "sum": logits.float().sum().item(),
                "first_5": logits.flatten()[:5].tolist(),
                "values": logits.flatten().tolist(),
            },
            "layer_0_output": results["prefill"]["layers.0.hidden.output"],
            **results["prefill"]
        },
        "decode_expected": {
            "logits": tensor_stats(decode_logits, include_values=True),
            "last_hidden_state": tensor_stats(decode_outputs.last_hidden_state),
            "cache_len_before": seq_len,
            "cache_len_after": seq_len + 1,
            "layer_0_output": results["decode"]["layers.0.hidden.output"],
            **results["decode"],
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(reference_data, f, indent=2)
    print(f"Reference data saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="reference.json")
    args = parser.parse_args()
    generate_reference(args.model_dir, args.output)
