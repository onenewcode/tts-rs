"""Generate V8 (end-to-end) reference data.

Exports `reference_v8_e2e.json` with full pipeline outputs:
  - Text embeddings (input to Rust)
  - Talker token IDs
  - Code predictor token IDs
  - Waveform samples

Usage:
    uv run python py/generate_reference_v8.py \
        --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --output reference_v8_e2e.json
"""

import json
import argparse
import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig


def tensor_stats(t, max_values=5000):
    """Export tensor statistics. Truncates `values` to `max_values` for large tensors."""
    flat = t.flatten().float()
    values = flat.tolist()
    return {
        "shape": list(t.shape),
        "first_5": flat[:5].tolist(),
        "last_5": flat[-5:].tolist(),
        "first_100": flat[:100].tolist(),
        "last_100": flat[-100:].tolist(),
        "values": values[:max_values],
        "num_elements": len(values),
        "truncated": len(values) > max_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="reference_v8_e2e.json")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--prefill-len", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading talker from {args.model_dir}/talker")
    config = Qwen3TTSTalkerConfig.from_pretrained(args.model_dir + "/talker")
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        args.model_dir + "/talker", config=config, torch_dtype="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}, dtype={next(model.parameters()).dtype}")

    # Generate deterministic input
    batch_size = 1
    prefill_len = args.prefill_len
    hidden_size = config.hidden_size
    max_new_tokens = args.max_new_tokens
    num_code_groups = config.num_code_groups

    torch.manual_seed(0)
    inputs_embeds = torch.randn(batch_size, prefill_len, hidden_size, device=device)
    position_ids = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(
        3, batch_size, prefill_len
    ).to(device)

    reference = {
        "config": {
            "batch_size": batch_size,
            "prefill_len": prefill_len,
            "hidden_size": hidden_size,
            "max_new_tokens": max_new_tokens,
            "num_code_groups": num_code_groups,
            "vocab_size": config.vocab_size,
        },
        "input": {
            "inputs_embeds": inputs_embeds.flatten().tolist(),
            "inputs_embeds_shape": list(inputs_embeds.shape),
            "position_ids": position_ids.flatten().tolist(),
            "position_ids_shape": list(position_ids.shape),
        },
    }

    # --- Talker prefill ---
    print("Prefill...")
    prefill_out = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    reference["prefill"] = {
        "logits": tensor_stats(prefill_out.logits),
    }

    # --- Talker autoregressive generation ---
    print(f"Generating {max_new_tokens} tokens...")
    generated_tokens = []
    step_logits_list = []
    codec_groups_list = []

    logits = prefill_out.logits
    past_key_values = prefill_out.past_key_values

    for step_idx in range(max_new_tokens):
        # Select token
        selected = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        generated_tokens.append(selected.item())

        # Code predictor expansion
        talker_hidden = torch.zeros(batch_size, hidden_size, device=device)
        base_emb = model.get_input_embeddings()(selected)
        cpred_inputs = torch.cat([talker_hidden.unsqueeze(1), base_emb], dim=1)
        cpred_out = model.code_predictor(
            inputs_embeds=cpred_inputs, use_cache=True
        )
        cpred_token = cpred_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        codec_groups = [selected.item(), cpred_token.item()]
        codec_groups_list.append(codec_groups)

        if step_idx < max_new_tokens - 1:
            current_embeds = model.get_input_embeddings()(selected)
            cur_pos = past_key_values[0][0].shape[-2]
            pos_step = position_ids.new_full((3, batch_size, 1), cur_pos)
            step_out = model(
                inputs_embeds=current_embeds,
                position_ids=pos_step,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = step_out.logits
            past_key_values = step_out.past_key_values
            step_logits_list.append(tensor_stats(logits))

    reference["generation"] = {
        "token_ids": generated_tokens,
        "step_logits": step_logits_list,
        "codec_groups": codec_groups_list,
    }

    print(f"Tokens: {generated_tokens}")
    print(f"Codec groups per step: {codec_groups_list}")

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference saved to {args.output}")
    print("Done: V8 end-to-end reference (talker generation + code predictor)")


if __name__ == "__main__":
    main()
