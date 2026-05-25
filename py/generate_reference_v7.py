"""Generate V7 (speech tokenizer decoder) reference data.

Exports `reference_v7_decoder.json` with decoder activations and waveform statistics.

Usage:
    uv run python py/generate_reference_v7.py \
        --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --output reference_v7_decoder.json
"""

import json
import argparse
import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSTalkerConfig,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def tensor_stats(t):
    """Export tensor metadata plus full flattened values."""
    flat = t.flatten().float()
    return {
        "shape": list(t.shape),
        "first_5": flat[:5].tolist(),
        "last_5": flat[-5:].tolist(),
        "first_100": flat[:100].tolist(),
        "last_100": flat[-100:].tolist(),
        "values": flat.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="reference_v7_decoder.json")
    args = parser.parse_args()

    print(f"Loading models from {args.model_dir}")

    # Load full TTS model (talker + speech tokenizer)
    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map="auto",
        dtype="auto",
    )
    model.eval()

    # Generate some codec tokens via the talker (small test case)
    talker = model.talker
    batch_size = 1
    prefill_len = 5
    max_new_tokens = 3
    hidden_size = talker.config.hidden_size

    torch.manual_seed(0)
    inputs_embeds = torch.randn(batch_size, prefill_len, hidden_size)
    position_ids = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(
        3, batch_size, prefill_len
    )

    # Greedy talker generation to produce codec token IDs
    prefill_out = talker(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = prefill_out.logits
    past_key_values = prefill_out.past_key_values
    first_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    generated = [first_token]

    for _ in range(1, max_new_tokens):
        current_embeds = talker.get_input_embeddings()(first_token)
        cur_pos = past_key_values[0][0].shape[-2]
        position_ids_step = position_ids.new_full((3, batch_size, 1), cur_pos)
        step_out = talker(
            inputs_embeds=current_embeds,
            position_ids=position_ids_step,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = step_out.logits
        past_key_values = step_out.past_key_values
        first_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        generated.append(first_token)

    codec_ids = torch.cat(generated, dim=1)  # [1, max_new_tokens]
    print(f"Generated talker tokens: shape={codec_ids.shape}, values={codec_ids.tolist()}")

    # Now run code predictor expansion per time step
    all_codec_groups = []
    for t in range(max_new_tokens):
        main_token = codec_ids[:, t:t + 1]  # [1, 1]
        # Run code predictor (simplified - using teacher-forced approach)
        base_emb = talker.get_input_embeddings()(main_token)
        h_state = torch.zeros(batch_size, hidden_size)  # placeholder hidden state
        cpred_inputs = torch.cat([h_state.unsqueeze(1), base_emb], dim=1)
        cpred_out = talker.code_predictor(inputs_embeds=cpred_inputs, use_cache=True)
        cpred_token = cpred_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        # For simplicity, repeat first token for all codec groups
        # (full code predictor autoregression omitted for brevity)
        group_tokens = torch.cat([main_token, cpred_token], dim=1)  # [1, 2]
        all_codec_groups.append(group_tokens)

    # Stack: [batch, num_groups, time_steps]
    codec_3d = torch.stack([g.unsqueeze(-1) for g in all_codec_groups], dim=-1)
    codec_3d = codec_3d.squeeze(0)  # [num_groups, time_steps]
    print(f"Codec 3D shape: {codec_3d.shape}")

    # Run speech tokenizer decoder
    print("Running speech tokenizer decoder...")
    with torch.no_grad():
        # Access the internal speech tokenizer decoder
        st = model.speech_tokenizer
        # The decoder is part of the speech tokenizer model
        # We need to trace through the model to find decoder outputs
        decoder_out = st.decode_tokens(codec_3d.unsqueeze(0))  # [1, 1, samples]

    samples = decoder_out.shape[-1]
    print(f"Decoder output: shape={decoder_out.shape}, samples={samples}")

    reference = {
        "input": {
            "codec_ids": codec_ids.tolist(),
            "codec_3d_shape": list(codec_3d.shape),
            "codec_3d_values": codec_3d.flatten().tolist(),
        },
        "expected": {
            "waveform": tensor_stats(decoder_out),
            "num_samples": samples,
        },
    }

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference saved to {args.output}")
    print("Done: V7 decoder reference")


if __name__ == "__main__":
    main()
