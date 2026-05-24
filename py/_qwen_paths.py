from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = REPO_ROOT / "Qwen"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "qwen3_tts"


def find_qwen_model_dir() -> Path:
    for path in sorted(QWEN_ROOT.iterdir()):
        if path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            return path
    raise FileNotFoundError(f"no qwen model directory found under {QWEN_ROOT}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def component_artifact_dir(component: str) -> Path:
    path = ARTIFACT_ROOT / component
    path.mkdir(parents=True, exist_ok=True)
    return path
