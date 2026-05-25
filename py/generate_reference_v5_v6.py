"""Generate V5 (sampling controls) and V6 (repetition penalty) reference data.

Exports `reference_v5_sampling.json` and `reference_v6_penalty.json`.

Usage:
    uv run python py/generate_reference_v5_v6.py \
        --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --output reference_v5_v6.json
"""

import json
import argparse
import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration


def tensor_stats(t):
    """Export tensor metadata plus full flattened values for element-wise comparison."""
    flat = t.flatten().float()
    return {
        "shape": list(t.shape),
        "first_5": flat[:5].tolist(),
        "last_5": flat[-5:].tolist(),
        "values": flat.tolist(),
    }


def run_greedy_generation(model, inputs_embeds, position_ids, max_new_tokens):
    """Deterministic greedy generation — V3 baseline, reused as V5/V6 control."""
    device = inputs_embeds.device
    batch_size, prefill_len, hidden_size = inputs_embeds.shape

    prefill_out = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = prefill_out.logits
    past_key_values = prefill_out.past_key_values

    selected = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)  # [batch, 1]
    generated = [selected]
    step_logits = []

    for _ in range(1, max_new_tokens):
        current_embeds = model.get_input_embeddings()(selected)
        position_ids_step = position_ids.new_full(
            (3, batch_size, 1), past_key_values[0][0].shape[-2]
        )
        step_out = model(
            inputs_embeds=current_embeds,
            position_ids=position_ids_step,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = step_out.logits
        past_key_values = step_out.past_key_values
        selected = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        generated.append(selected)
        step_logits.append(tensor_stats(logits))

    token_ids = torch.cat(generated, dim=1)
    return token_ids, step_logits, prefill_out


def run_sampling_generation(model, inputs_embeds, position_ids, max_new_tokens,
                            temperature=0.9, top_k=50, top_p=1.0):
    """V5: Stochastic sampling with temperature, top-k, top-p."""
    import torch.nn.functional as F

    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]
    torch.manual_seed(42)  # reproducible

    prefill_out = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = prefill_out.logits
    past_key_values = prefill_out.past_key_values

    def sample_token(lgts):
        """Apply temperature → top-k → top-p → softmax → multinomial."""
        lgts = lgts[:, -1, :] / max(temperature, 1e-5)  # [batch, vocab]
        if top_k and top_k > 0:
            kth = lgts.topk(top_k, dim=-1).values[:, -1:]  # [batch, 1]
            lgts[lgts < kth] = float("-inf")
        if top_p < 1.0:
            sorted_lgts, sorted_idx = lgts.sort(dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_lgts, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            keep = (cumsum - sorted_probs) < top_p
            sorted_lgts[~keep] = float("-inf")
            # Restore original order
            lgts = sorted_lgts.new_full(lgts.shape, float("-inf"))
            lgts = lgts.scatter(1, sorted_idx, sorted_lgts)
        probs = F.softmax(lgts.float(), dim=-1)
        return torch.multinomial(probs, 1)  # [batch, 1]

    selected = sample_token(logits)
    generated = [selected]
    step_logits = []

    for _ in range(1, max_new_tokens):
        current_embeds = model.get_input_embeddings()(selected)
        position_ids_step = position_ids.new_full(
            (3, batch_size, 1), past_key_values[0][0].shape[-2]
        )
        step_out = model(
            inputs_embeds=current_embeds,
            position_ids=position_ids_step,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = step_out.logits
        past_key_values = step_out.past_key_values
        selected = sample_token(logits)
        generated.append(selected)
        step_logits.append(tensor_stats(logits))

    token_ids = torch.cat(generated, dim=1)
    return token_ids, step_logits, prefill_out


def run_repetition_penalty_generation(model, inputs_embeds, position_ids,
                                      max_new_tokens, penalty=1.2):
    """V6: Greedy generation with repetition penalty applied.

    For each past token t: logits[:, t] /= penalty
    """
    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]

    prefill_out = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = prefill_out.logits
    past_key_values = prefill_out.past_key_values

    generated = []
    step_logits = []
    all_token_ids = []

    for step_idx in range(max_new_tokens):
        # Apply repetition penalty to last-position logits
        last_lgts = logits[:, -1, :].clone()
        for t in all_token_ids:
            last_lgts[0, t] /= penalty
        selected = last_lgts.argmax(dim=-1).unsqueeze(-1)
        generated.append(selected)
        all_token_ids.append(selected.item())

        if step_idx < max_new_tokens - 1:
            current_embeds = model.get_input_embeddings()(selected)
            position_ids_step = position_ids.new_full(
                (3, batch_size, 1), past_key_values[0][0].shape[-2]
            )
            step_out = model(
                inputs_embeds=current_embeds,
                position_ids=position_ids_step,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = step_out.logits
            past_key_values = step_out.past_key_values
            step_logits.append(tensor_stats(step_out.logits))

    token_ids = torch.cat(generated, dim=1)
    return token_ids, step_logits, prefill_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="reference_v5_v6.json")
    args = parser.parse_args()

    print(f"Loading talker from {args.model_dir}/talker")
    config = Qwen3TTSTalkerConfig.from_pretrained(args.model_dir + "/talker")
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        args.model_dir + "/talker", config=config, torch_dtype="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}, dtype={next(model.parameters()).dtype}")

    # Shared input
    batch_size = 1
    prefill_len = 5
    hidden_size = config.hidden_size
    max_new_tokens = 10

    torch.manual_seed(0)
    inputs_embeds = torch.randn(batch_size, prefill_len, hidden_size, device=device)
    position_ids = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(
        3, batch_size, prefill_len
    ).to(device)

    reference = {}

    # --- V3: Greedy baseline (control) ---
    print("V3 greedy baseline...")
    greedy_tokens, greedy_step_logits, _ = run_greedy_generation(
        model, inputs_embeds.clone(), position_ids.clone(), max_new_tokens
    )
    reference["v3_greedy"] = {
        "token_ids": greedy_tokens.tolist(),
        "step_logits": greedy_step_logits,
    }

    # --- V5: Sampling ---
    print("V5 sampling (temperature=0.9, top_k=50)...")
    torch.manual_seed(42)
    sample_tokens, sample_step_logits, _ = run_sampling_generation(
        model, inputs_embeds.clone(), position_ids.clone(), max_new_tokens,
        temperature=0.9, top_k=50, top_p=0.95,
    )
    reference["v5_sampling"] = {
        "config": {"temperature": 0.9, "top_k": 50, "top_p": 0.95, "seed": 42},
        "token_ids": sample_tokens.tolist(),
        "step_logits": sample_step_logits,
    }

    # --- V5: Sampling (top-k=1 = approximate greedy) ---
    print("V5 sampling (top_k=1, temperature=1e-5 = near-greedy)...")
    torch.manual_seed(42)
    near_greedy_tokens, near_greedy_logits, _ = run_sampling_generation(
        model, inputs_embeds.clone(), position_ids.clone(), max_new_tokens,
        temperature=1e-5, top_k=1, top_p=1.0,
    )
    reference["v5_near_greedy"] = {
        "config": {"temperature": 1e-5, "top_k": 1, "top_p": 1.0, "seed": 42},
        "token_ids": near_greedy_tokens.tolist(),
        "step_logits": near_greedy_logits,
    }

    # --- V6: Repetition penalty ---
    print("V6 repetition_penalty=1.2...")
    torch.manual_seed(0)
    rp_tokens, rp_step_logits, _ = run_repetition_penalty_generation(
        model, inputs_embeds.clone(), position_ids.clone(), max_new_tokens,
        penalty=1.2,
    )
    reference["v6_penalty_12"] = {
        "penalty": 1.2,
        "token_ids": rp_tokens.tolist(),
        "step_logits": rp_step_logits,
    }

    print("V6 repetition_penalty=1.0 (no-op, should match V3 greedy)...")
    torch.manual_seed(0)
    rp_off_tokens, rp_off_logits, _ = run_repetition_penalty_generation(
        model, inputs_embeds.clone(), position_ids.clone(), max_new_tokens,
        penalty=1.0,
    )
    reference["v6_penalty_off"] = {
        "penalty": 1.0,
        "token_ids": rp_off_tokens.tolist(),
        "step_logits": rp_off_logits,
    }

    # Verify V6 no-op matches V3 greedy
    assert greedy_tokens.tolist() == rp_off_tokens.tolist(), \
        "V6 penalty=1.0 should match V3 greedy tokens!"

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference saved to {args.output}")
    print("Done: V3 greedy + V5 sampling + V5 near-greedy + V6 penalty")


if __name__ == "__main__":
    main()
