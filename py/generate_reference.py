import torch
import json
import argparse
import os
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

def generate_reference(model_dir, output_json, input_text="Hello"):
    # 1. Load Model
    config = Qwen3TTSTalkerConfig.from_pretrained(model_dir)
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(model_dir).eval()
    
    # 2. Prepare dummy but deterministic input
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(3, batch_size, 1)
    
    # 3. Inference with hooks to catch layer outputs
    results = {}
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            results[name] = {
                "shape": list(output.shape),
                "sum": output.sum().item(),
                "mean": output.mean().item(),
                "first_5": output.flatten()[:5].tolist()
            }
        return fn

    # Register hooks for key layers
    model.model.layers[0].register_forward_hook(hook_fn("layer_0_output"))
    model.model.norm.register_forward_hook(hook_fn("final_norm"))

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = outputs.logits
        
    # 4. Save to JSON
    reference_data = {
        "input": {
            "inputs_embeds": inputs_embeds.tolist(),
            "position_ids": position_ids.tolist(),
        },
        "expected": {
            "logits": {
                "shape": list(logits.shape),
                "sum": logits.sum().item(),
                "first_5": logits.flatten()[:5].tolist()
            },
            **results
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
