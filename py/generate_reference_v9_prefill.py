"""Generate V9 CustomVoice prefill alignment reference data."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel


def tensor_stats(tensor, max_values=None):
    flat = tensor.detach().cpu().float().flatten()
    values = flat.tolist()
    if max_values is not None:
        values = values[:max_values]
    return {
        "shape": list(tensor.shape),
        "values": values,
        "num_elements": flat.numel(),
        "truncated": max_values is not None and flat.numel() > max_values,
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
    }


def build_codec_prefix_ids(config, language, speaker):
    talker = config.talker_config
    language_key = (language or "Auto").lower()
    speaker_key = speaker.lower() if speaker else None
    if language_key == "auto":
        language_id = None
    else:
        language_id = talker.codec_language_id[language_key]
    if language_key in ("chinese", "auto") and speaker_key:
        dialect = talker.spk_is_dialect[speaker_key]
        if dialect is not False:
            language_id = talker.codec_language_id[dialect]
    if language_id is None:
        prefix = [
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
        ]
    else:
        prefix = [
            talker.codec_think_id,
            talker.codec_think_bos_id,
            language_id,
            talker.codec_think_eos_id,
        ]
    if speaker_key:
        prefix.append(talker.spk_id[speaker_key])
    prefix.extend([talker.codec_pad_id, talker.codec_bos_id])
    return prefix


def load_model(model_dir):
    try:
        wrapper = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", dtype=torch.bfloat16
        )
    except TypeError:
        wrapper = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", torch_dtype=torch.bfloat16
        )
    force_eager_attention(wrapper)
    return wrapper


def force_eager_attention(wrapper):
    model = getattr(wrapper, "model", None)
    talker = getattr(model, "talker", None)
    code_predictor = getattr(talker, "code_predictor", None)
    candidates = [
        getattr(wrapper, "config", None),
        getattr(model, "config", None),
        getattr(talker, "config", None),
        getattr(getattr(talker, "model", None), "config", None),
        getattr(code_predictor, "config", None),
        getattr(getattr(code_predictor, "model", None), "config", None),
    ]
    for config in candidates:
        if config is not None:
            config._attn_implementation = "eager"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    wrapper = load_model(model_dir)
    wrapper.model.eval()
    prompt = wrapper._build_assistant_text(args.text)
    input_ids = wrapper._tokenize_texts([prompt])
    captured = {}

    def capture_generate(**kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().cpu()
        captured["attention_mask"] = kwargs["attention_mask"].detach().cpu()
        captured["trailing_text_hidden"] = kwargs["trailing_text_hidden"].detach().cpu()
        captured["tts_pad_embed"] = kwargs["tts_pad_embed"].detach().cpu()
        raise RuntimeError("__captured_prefill__")

    original_generate = wrapper.model.talker.generate
    wrapper.model.talker.generate = capture_generate
    try:
        wrapper.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=True,
            do_sample=False,
            max_new_tokens=2,
            repetition_penalty=1.0,
        )
    except RuntimeError as exc:
        if str(exc) != "__captured_prefill__":
            raise
    finally:
        wrapper.model.talker.generate = original_generate

    token_ids = input_ids[0].detach().cpu().reshape(-1).tolist()
    attention_mask = captured["attention_mask"].to(torch.int64)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": prompt,
                "text": args.text,
                "language": args.language,
                "speaker": args.speaker,
                "text_token_ids": token_ids,
                "codec_prefix_ids": build_codec_prefix_ids(
                    wrapper.model.config, args.language, args.speaker
                ),
                "attention_mask": attention_mask.tolist(),
                "position_ids": position_ids.tolist(),
                "inputs_embeds": tensor_stats(captured["inputs_embeds"]),
                "trailing_text_hidden": tensor_stats(captured["trailing_text_hidden"]),
                "tts_pad_embed": tensor_stats(captured["tts_pad_embed"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
