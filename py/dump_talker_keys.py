from __future__ import annotations

import hashlib

import torch

from safetensors import safe_open

from _qwen_paths import component_artifact_dir, find_qwen_model_dir, write_json


def canonical_dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
        torch.bool: "Bool",
    }
    return names.get(dtype, str(dtype))


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    model_dir = find_qwen_model_dir()
    weights_path = model_dir / "model.safetensors"
    output_path = component_artifact_dir("talker") / "python_source_manifest.json"

    entries: list[dict[str, object]] = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            tensor = handle.get_tensor(key)
            entries.append(
                {
                    "path": key,
                    "shape": list(tensor.shape),
                    "dtype": canonical_dtype_name(tensor.dtype),
                    "sha256": tensor_sha256(tensor),
                }
            )

    payload = {
        "tensor_count": len(entries),
        "entries": entries,
    }
    write_json(output_path, payload)

    print(f"model_dir: {model_dir}")
    print(f"weights: {weights_path}")
    print(f"manifest: {output_path}")
    print(f"tensor_count: {len(entries)}")


if __name__ == "__main__":
    main()
