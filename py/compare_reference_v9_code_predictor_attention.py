"""Compare two eager code-predictor reference runs."""

import argparse
import json
from pathlib import Path

import torch

from generate_reference_v9_code_predictor import (
    install_code_predictor_capture,
    load_model,
    predictor_reference,
)


REPORT_TOLERANCE = 1e-3


def activation_order_key(name):
    if not name.startswith("layers."):
        return (10_000, 10_000, name)
    parts = name.split(".", 2)
    if len(parts) != 3:
        return (10_000, 10_000, name)
    try:
        layer = int(parts[1])
    except ValueError:
        layer = 10_000
    stage = {
        "input_norm.output": 0,
        "q_proj.output": 1,
        "k_proj.output": 2,
        "v_proj.output": 3,
        "q_norm.output": 4,
        "k_norm.output": 5,
        "q_rot.output": 6,
        "k_rot.output": 7,
        "attn.weights": 8,
        "attn.output": 9,
        "attn_residual.output": 10,
        "post_attention_norm.output": 11,
        "mlp.gate": 12,
        "mlp.up": 13,
        "mlp.activated_gate": 14,
        "mlp.product": 15,
        "mlp.output": 16,
        "hidden.output": 17,
    }.get(parts[2], 10_000)
    return (layer, stage, parts[2])


def max_abs_summary(left, right):
    left_values = torch.tensor(left["values"], dtype=torch.float32)
    right_values = torch.tensor(right["values"], dtype=torch.float32)
    if left["shape"] != right["shape"]:
        return {
            "shape_mismatch": True,
            "left_shape": left["shape"],
            "right_shape": right["shape"],
        }
    diff = (left_values - right_values).abs()
    max_abs, max_idx = torch.max(diff, dim=0)
    max_idx = int(max_idx.item())
    return {
        "shape_mismatch": False,
        "max_abs": float(max_abs.item()),
        "max_idx": max_idx,
        "exceed_count": int((diff > REPORT_TOLERANCE).sum().item()),
        "default_value": float(left_values[max_idx].item()),
        "eager_value": float(right_values[max_idx].item()),
    }


def compare_step(baseline_step, eager_step):
    score_summaries = [
        max_abs_summary(baseline_score, eager_score)
        for baseline_score, eager_score in zip(baseline_step["scores"], eager_step["scores"])
    ]
    activation_summaries = {}
    for head_idx, baseline_head in baseline_step["activations"].items():
        eager_head = eager_step["activations"].get(head_idx, {})
        common_names = sorted(set(baseline_head) & set(eager_head), key=activation_order_key)
        summaries = [
            (name, max_abs_summary(baseline_head[name], eager_head[name]))
            for name in common_names
        ]
        first = [
            {"name": name, **summary}
            for name, summary in summaries
            if not summary.get("shape_mismatch") and summary["exceed_count"] > 0
        ][:8]
        top = sorted(
            [
                {"name": name, **summary}
                for name, summary in summaries
                if not summary.get("shape_mismatch") and summary["exceed_count"] > 0
            ],
            key=lambda item: item["max_abs"],
            reverse=True,
        )[:8]
        if first or top:
            activation_summaries[head_idx] = {"first": first, "top": top}

    return {
        "step_idx": baseline_step["step_idx"],
        "base_token_id": baseline_step["base_token_id"],
        "baseline_codec_groups": baseline_step["expected_codec_groups"],
        "eager_codec_groups": eager_step["expected_codec_groups"],
        "groups_match": baseline_step["expected_codec_groups"]
        == eager_step["expected_codec_groups"],
        "score_summaries": score_summaries,
        "activation_summaries": activation_summaries,
    }


@torch.no_grad()
def run_reference(wrapper, codes, hidden_states, steps):
    captured = install_code_predictor_capture(wrapper.model.talker.code_predictor)
    return [
        predictor_reference(wrapper, codes, hidden_states, step_idx, captured)
        for step_idx in steps
        if 0 <= step_idx < codes.shape[0]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    parser.add_argument("--steps", default="2")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    requested_steps = [int(item) for item in args.steps.split(",") if item]

    baseline_wrapper = load_model(model_dir)
    baseline_wrapper.model.eval()
    input_ids = baseline_wrapper._tokenize_texts(
        [baseline_wrapper._build_assistant_text(args.text)]
    )
    codes_list, hidden_states_list = baseline_wrapper.model.generate(
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

    baseline_steps = run_reference(baseline_wrapper, codes, hidden_states, requested_steps)

    eager_wrapper = load_model(model_dir)
    eager_wrapper.model.eval()
    eager_wrapper.model.talker.code_predictor.config._attn_implementation = "eager"
    eager_wrapper.model.talker.code_predictor.model.config._attn_implementation = "eager"
    eager_steps = run_reference(eager_wrapper, codes, hidden_states, requested_steps)

    output = {
        "report_tolerance": REPORT_TOLERANCE,
        "steps": [
            compare_step(baseline_step, eager_step)
            for baseline_step, eager_step in zip(baseline_steps, eager_steps)
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
