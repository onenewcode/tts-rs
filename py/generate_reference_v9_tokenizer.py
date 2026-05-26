"""Generate V9 tokenizer alignment reference data."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompt = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "text": args.text,
                "prompt": prompt,
                "token_ids": token_ids,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
