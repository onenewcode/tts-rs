"""Placeholder V9 E2E oracle entry point.

The ignored Rust E2E alignment test uses this script as the stable command surface for
future full-checkpoint comparison of talker tokens, code predictor groups, and waveform
previews.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model_dir": args.model_dir, "status": "placeholder"}, f, indent=2)


if __name__ == "__main__":
    main()
