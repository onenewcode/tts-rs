"""Generate V9 end-to-end alignment reference data."""

import argparse
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel


def tensor_preview(tensor, max_values=256):
    flat = tensor.detach().cpu().float().flatten()
    return {
        "shape": list(tensor.shape),
        "values": flat[:max_values].tolist(),
        "num_elements": flat.numel(),
        "truncated": flat.numel() > max_values,
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
    }


def score_topk(scores, k=5):
    topk = []
    for score in scores:
        values, indices = torch.topk(score.detach().cpu().float(), k=min(k, score.shape[-1]), dim=-1)
        topk.append(
            {
                "ids": indices[0].to(torch.int64).tolist(),
                "values": values[0].tolist(),
            }
        )
    return topk


def load_model(model_dir):
    try:
        return Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", dtype=torch.bfloat16
        )
    except TypeError:
        return Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map="cpu", torch_dtype=torch.bfloat16
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    wrapper = load_model(model_dir)
    wrapper.model.eval()
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(args.text)])
    with torch.no_grad():
        codes_list, hidden_states_list = wrapper.model.generate(
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
        wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
            [{"audio_codes": codes_list[0]}]
        )
        manual_predictor_groups = []
        manual_predictor_topk = []
        for step_idx in range(codes_list[0].shape[0]):
            base_id = codes_list[0][step_idx : step_idx + 1, 0:1].to(torch.long)
            talker_hidden = hidden_states_list[0][step_idx : step_idx + 1].unsqueeze(1)
            base_embed = wrapper.model.talker.get_input_embeddings()(base_id)
            predictor = wrapper.model.talker.code_predictor.generate(
                inputs_embeds=torch.cat((talker_hidden, base_embed), dim=1),
                max_new_tokens=wrapper.model.talker.config.num_code_groups - 1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            manual_predictor_groups.append(
                torch.cat((base_id, predictor.sequences), dim=-1)
                .detach()
                .cpu()
                .to(torch.int64)
                .squeeze(0)
                .tolist()
            )
            manual_predictor_topk.append(score_topk(predictor.scores))

    codes = codes_list[0].detach().cpu().to(torch.int64)
    hidden = hidden_states_list[0].detach().cpu()
    waveform = torch.tensor(wavs[0], dtype=torch.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": args.text,
                "language": args.language,
                "speaker": args.speaker,
                "max_new_tokens": args.max_new_tokens,
                "sample_rate": sample_rate,
                "base_token_ids": codes[:, 0].tolist(),
                "codec_groups": codes.tolist(),
                "manual_predictor_groups": manual_predictor_groups,
                "manual_predictor_topk": manual_predictor_topk,
                "codec_shape": list(codes.unsqueeze(0).transpose(1, 2).shape),
                "talker_hidden": tensor_preview(hidden),
                "waveform": tensor_preview(waveform),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
