from __future__ import annotations

from safetensors import safe_open

from _qwen_paths import component_artifact_dir, find_qwen_model_dir, load_json, write_json


def expected_talker_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    talker = config["talker_config"]
    predictor = talker["code_predictor_config"]
    expected: dict[str, tuple[int, ...]] = {}

    hidden_size = talker["hidden_size"]
    intermediate_size = talker["intermediate_size"]
    num_layers = talker["num_hidden_layers"]
    num_heads = talker["num_attention_heads"]
    num_kv_heads = talker["num_key_value_heads"]
    head_dim = talker["head_dim"]
    vocab_size = talker["vocab_size"]
    text_hidden_size = talker["text_hidden_size"]
    text_vocab_size = talker["text_vocab_size"]

    expected["talker.model.codec_embedding.weight"] = (vocab_size, hidden_size)
    expected["talker.model.norm.weight"] = (hidden_size,)
    expected["talker.model.text_embedding.weight"] = (text_vocab_size, text_hidden_size)
    expected["talker.text_projection.linear_fc1.weight"] = (text_hidden_size, text_hidden_size)
    expected["talker.text_projection.linear_fc1.bias"] = (text_hidden_size,)
    expected["talker.text_projection.linear_fc2.weight"] = (hidden_size, text_hidden_size)
    expected["talker.text_projection.linear_fc2.bias"] = (hidden_size,)
    expected["talker.codec_head.weight"] = (vocab_size, hidden_size)

    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
    for layer_idx in range(num_layers):
        prefix = f"talker.model.layers.{layer_idx}"
        expected[f"{prefix}.input_layernorm.weight"] = (hidden_size,)
        expected[f"{prefix}.post_attention_layernorm.weight"] = (hidden_size,)
        expected[f"{prefix}.mlp.gate_proj.weight"] = (intermediate_size, hidden_size)
        expected[f"{prefix}.mlp.up_proj.weight"] = (intermediate_size, hidden_size)
        expected[f"{prefix}.mlp.down_proj.weight"] = (hidden_size, intermediate_size)
        expected[f"{prefix}.self_attn.q_proj.weight"] = (q_out, hidden_size)
        expected[f"{prefix}.self_attn.k_proj.weight"] = (kv_out, hidden_size)
        expected[f"{prefix}.self_attn.v_proj.weight"] = (kv_out, hidden_size)
        expected[f"{prefix}.self_attn.o_proj.weight"] = (hidden_size, q_out)
        expected[f"{prefix}.self_attn.q_norm.weight"] = (head_dim,)
        expected[f"{prefix}.self_attn.k_norm.weight"] = (head_dim,)

    predictor_hidden = predictor["hidden_size"]
    predictor_intermediate = predictor["intermediate_size"]
    predictor_layers = predictor["num_hidden_layers"]
    predictor_heads = predictor["num_attention_heads"]
    predictor_kv_heads = predictor["num_key_value_heads"]
    predictor_head_dim = predictor["head_dim"]
    predictor_vocab = predictor["vocab_size"]
    predictor_groups = predictor["num_code_groups"] - 1

    predictor_q_out = predictor_heads * predictor_head_dim
    predictor_kv_out = predictor_kv_heads * predictor_head_dim

    expected["talker.code_predictor.model.norm.weight"] = (predictor_hidden,)
    for group_idx in range(predictor_groups):
        expected[f"talker.code_predictor.model.codec_embedding.{group_idx}.weight"] = (
            predictor_vocab,
            hidden_size,
        )
        expected[f"talker.code_predictor.lm_head.{group_idx}.weight"] = (
            predictor_vocab,
            predictor_hidden,
        )

    for layer_idx in range(predictor_layers):
        prefix = f"talker.code_predictor.model.layers.{layer_idx}"
        expected[f"{prefix}.input_layernorm.weight"] = (predictor_hidden,)
        expected[f"{prefix}.post_attention_layernorm.weight"] = (predictor_hidden,)
        expected[f"{prefix}.mlp.gate_proj.weight"] = (predictor_intermediate, predictor_hidden)
        expected[f"{prefix}.mlp.up_proj.weight"] = (predictor_intermediate, predictor_hidden)
        expected[f"{prefix}.mlp.down_proj.weight"] = (predictor_hidden, predictor_intermediate)
        expected[f"{prefix}.self_attn.q_proj.weight"] = (predictor_q_out, predictor_hidden)
        expected[f"{prefix}.self_attn.k_proj.weight"] = (predictor_kv_out, predictor_hidden)
        expected[f"{prefix}.self_attn.v_proj.weight"] = (predictor_kv_out, predictor_hidden)
        expected[f"{prefix}.self_attn.o_proj.weight"] = (predictor_hidden, predictor_q_out)
        expected[f"{prefix}.self_attn.q_norm.weight"] = (predictor_head_dim,)
        expected[f"{prefix}.self_attn.k_norm.weight"] = (predictor_head_dim,)

    if predictor_hidden != hidden_size:
        expected["talker.code_predictor.small_to_mtp_projection.weight"] = (
            predictor_hidden,
            hidden_size,
        )
        expected["talker.code_predictor.small_to_mtp_projection.bias"] = (predictor_hidden,)

    return expected


def main() -> None:
    model_dir = find_qwen_model_dir()
    config = load_json(model_dir / "config.json")
    expected = expected_talker_shapes(config)
    output_path = component_artifact_dir("talker") / "python_structure_report.json"

    actual: dict[str, tuple[int, ...]] = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            actual[key] = tuple(handle.get_tensor(key).shape)

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong_shape = sorted(
        key for key in set(expected) & set(actual) if expected[key] != actual[key]
    )

    payload = {
        "model_dir": str(model_dir),
        "expected_tensors": len(expected),
        "actual_tensors": len(actual),
        "missing": missing,
        "unexpected": unexpected,
        "wrong_shape": [
            {
                "path": key,
                "expected": list(expected[key]),
                "actual": list(actual[key]),
            }
            for key in wrong_shape
        ],
        "exact_match": not (missing or unexpected or wrong_shape),
    }
    write_json(output_path, payload)

    print(f"model_dir: {model_dir}")
    print(f"report: {output_path}")
    print(f"expected_tensors: {len(expected)}")
    print(f"actual_tensors: {len(actual)}")
    print(f"missing: {len(missing)}")
    print(f"unexpected: {len(unexpected)}")
    print(f"wrong_shape: {len(wrong_shape)}")

    if missing or unexpected or wrong_shape:
        raise SystemExit(1)

    print("talker structure matches the checkpoint exactly")


if __name__ == "__main__":
    main()
