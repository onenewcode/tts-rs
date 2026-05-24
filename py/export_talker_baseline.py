from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch

from _qwen_paths import component_artifact_dir, find_qwen_model_dir, write_json
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


ATOL = 1e-3
RTOL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["all", "prefill_small_seq", "subtalker_teacher_forced"],
        default="all",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    return parser.parse_args()


def baseline_case_dir(case_name: str) -> Path:
    return component_artifact_dir("talker") / "baseline" / case_name


def export_json(case_dir: Path, case_name: str, kind: str) -> None:
    write_json(
        case_dir / "case.json",
        {
            "case_name": case_name,
            "kind": kind,
            "atol": ATOL,
            "rtol": RTOL,
        },
    )


def save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {name: export_tensor(tensor) for name, tensor in tensors.items()})


def export_tensor(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "values": tensor.flatten().tolist(),
    }


def make_prefill_position_ids(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len)


def make_teacher_forced_position_ids(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)


def make_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, seq_len), device=device, dtype=torch.long)


def make_additive_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=attention_mask.device, dtype=dtype)
    upper = torch.triu(
        torch.ones((seq_len, seq_len), device=attention_mask.device, dtype=torch.bool),
        diagonal=1,
    )
    mask = mask.masked_fill(upper.view(1, 1, seq_len, seq_len), float("-inf"))
    padding = attention_mask[:, None, None, :] == 0
    return mask.masked_fill(padding, float("-inf"))


@torch.no_grad()
def collect_talker_prefill(
    talker,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    hidden_states = inputs_embeds
    causal_mask = make_additive_causal_mask(attention_mask, dtype=hidden_states.dtype)
    position_embeddings = talker.model.rotary_emb(hidden_states, position_ids)

    for layer_idx, layer in enumerate(talker.model.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attn_output, _ = layer.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=None,
        )
        hidden_states = residual + attn_output
        activations[f"layers.{layer_idx}.attn.output"] = attn_output

        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)
        mlp_output = layer.mlp(mlp_input)
        hidden_states = residual + mlp_output
        activations[f"layers.{layer_idx}.mlp.output"] = mlp_output
        activations[f"layers.{layer_idx}.hidden.output"] = hidden_states

    hidden_states = talker.model.norm(hidden_states)
    activations["model.norm.output"] = hidden_states
    logits = talker.codec_head(hidden_states)
    return activations, hidden_states, logits


@torch.no_grad()
def build_teacher_forced_inputs_embeds(
    talker,
    codec_ids: torch.Tensor,
    talker_hidden_states: torch.Tensor,
) -> torch.Tensor:
    inputs = [talker_hidden_states.unsqueeze(1)]
    for group_idx in range(talker.config.num_code_groups - 1):
        if group_idx == 0:
            inputs.append(talker.get_input_embeddings()(codec_ids[:, :1]))
        else:
            inputs.append(talker.code_predictor.get_input_embeddings()[group_idx - 1](codec_ids[:, group_idx : group_idx + 1]))
    return torch.cat(inputs, dim=1)


@torch.no_grad()
def collect_code_predictor_teacher_forced(
    talker,
    codec_ids: torch.Tensor,
    talker_hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    raw_inputs = build_teacher_forced_inputs_embeds(talker, codec_ids, talker_hidden_states)
    projected_inputs = talker.code_predictor.small_to_mtp_projection(raw_inputs)
    activations["code_predictor.input_embeds"] = projected_inputs

    hidden_states = projected_inputs
    causal_mask = make_additive_causal_mask(attention_mask, dtype=hidden_states.dtype)
    position_embeddings = talker.code_predictor.model.rotary_emb(hidden_states, position_ids)

    for layer_idx, layer in enumerate(talker.code_predictor.model.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attn_output, _ = layer.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=None,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)
        mlp_output = layer.mlp(mlp_input)
        hidden_states = residual + mlp_output
        activations[f"code_predictor.layers.{layer_idx}.hidden.output"] = hidden_states

    hidden_states = talker.code_predictor.model.norm(hidden_states)
    logits = torch.stack(
        [
            talker.code_predictor.lm_head[group_idx - 1](hidden_states[:, group_idx])
            for group_idx in range(1, talker.config.num_code_groups)
        ],
        dim=1,
    )
    activations["code_predictor.logits"] = logits
    return activations, logits


@torch.no_grad()
def export_prefill_case(talker, case_dir: Path) -> torch.Tensor:
    device = next(talker.parameters()).device
    codec_input_ids = torch.tensor([[1, 7, 13, 19]], device=device, dtype=torch.long) % talker.config.vocab_size
    text_token_ids = torch.tensor([[2, 5, 11, 17]], device=device, dtype=torch.long) % talker.config.text_vocab_size
    attention_mask = make_attention_mask(codec_input_ids.shape[1], device)
    position_ids = make_prefill_position_ids(codec_input_ids.shape[1], device)

    text_projection = talker.text_projection(talker.get_text_embeddings()(text_token_ids))
    codec_embedding = talker.get_input_embeddings()(codec_input_ids)
    inputs_embeds = text_projection + codec_embedding
    activations, last_hidden_state, logits = collect_talker_prefill(
        talker,
        inputs_embeds,
        position_ids,
        attention_mask,
    )

    export_json(case_dir, "prefill_small_seq", "prefill_small_seq")
    save_tensors(
        case_dir / "inputs.json",
        {
            "codec_input_ids": codec_input_ids,
            "text_token_ids": text_token_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        },
    )
    save_tensors(
        case_dir / "activations.json",
        {
            "text_projection.output": text_projection,
            "codec_embedding.output": codec_embedding,
            **activations,
        },
    )
    save_tensors(
        case_dir / "outputs.json",
        {
            "model.last_hidden_state": last_hidden_state,
            "codec_head.logits": logits,
        },
    )
    return last_hidden_state[:, -1, :].detach()


@torch.no_grad()
def export_teacher_forced_case(talker, case_dir: Path, talker_hidden_states: torch.Tensor) -> None:
    device = next(talker.parameters()).device
    codec_ids = (
        torch.arange(1, talker.config.num_code_groups + 1, device=device, dtype=torch.long)
        .view(1, talker.config.num_code_groups)
        % talker.config.vocab_size
    )
    attention_mask = make_attention_mask(talker.config.num_code_groups, device)
    position_ids = make_teacher_forced_position_ids(talker.config.num_code_groups, device)

    activations, logits = collect_code_predictor_teacher_forced(
        talker,
        codec_ids,
        talker_hidden_states,
        position_ids,
        attention_mask,
    )

    export_json(case_dir, "subtalker_teacher_forced", "subtalker_teacher_forced")
    save_tensors(
        case_dir / "inputs.json",
        {
            "codec_ids": codec_ids,
            "talker_hidden_states": talker_hidden_states,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        },
    )
    save_tensors(case_dir / "activations.json", activations)
    save_tensors(
        case_dir / "outputs.json",
        {
            "code_predictor.logits": logits,
        },
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or find_qwen_model_dir()
    model = Qwen3TTSForConditionalGeneration.from_pretrained(model_dir, torch_dtype="auto")
    model = model.to("cpu").eval()
    talker = model.talker.eval()

    prefill_hidden = None
    if args.case in {"all", "prefill_small_seq"}:
        prefill_hidden = export_prefill_case(talker, baseline_case_dir("prefill_small_seq"))
        print(f"exported: {baseline_case_dir('prefill_small_seq')}")

    if args.case in {"all", "subtalker_teacher_forced"}:
        if prefill_hidden is None:
            prefill_hidden = export_prefill_case(talker, baseline_case_dir("prefill_small_seq"))
        export_teacher_forced_case(
            talker,
            baseline_case_dir("subtalker_teacher_forced"),
            prefill_hidden,
        )
        print(f"exported: {baseline_case_dir('subtalker_teacher_forced')}")


if __name__ == "__main__":
    main()
