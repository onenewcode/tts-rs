# Qwen3-TTS V9 Alignment Debug Log

This log records every V9 Python-vs-Rust alignment investigation so future
work does not repeat the same probes. Heavy checks should be run in release
mode.

## Required Commands

```bash
cargo test --release -p tts_rs_qwen_burn
cargo test --release -p tts_rs_qwen_burn --test alignment_prefill -- --nocapture
cargo test --release -p tts_rs_qwen_burn --test alignment_talker_prefill -- --ignored --nocapture
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e -- --ignored --nocapture
```

The Python oracle may print SoX and flash-attn warnings. Those warnings are not
alignment failures.

## Current Conclusion

The current highest-priority failure is audio-codec parity, not only
talker/code-predictor drift. With Python eager codec groups as input, Rust
decoding no longer clips, but the waveform scale still differs materially from
Python:

- Command:
  `cargo test --release -p tts_rs_qwen_burn --test alignment_e2e rust_audio_codec_decodes_python_eager_codes_without_clipping -- --ignored --nocapture`
- Result on 2026-05-27:
  Rust `peak=0.15012264`, `rms=0.05151318`, `clip_fraction=0.0`;
  Python `peak=0.8203125`, `rms=0.09017588`, `clip_fraction=0.0`;
  waveform preview `max_abs=0.110171795`.
- Interpretation: the previous SnakeBeta/clamp fixes removed the 65x clipping
  failure, but decoder parity is still not solved. Do not use CLI
  `clip_fraction=0.0` or manifest `plausible` as proof of correct audio.
- Root-cause candidate found by comparing `/tmp/qwen3-tts-rs`:
  `TokenizerV2Decoder::forward` applies `pre_transformer` immediately after
  `pre_conv`, then runs `upsample_blocks`. The local Burn implementation was
  running `upsample` before `pre_transformer`. This is a structural decoder
  mismatch, not a precision issue. The local order has been changed to
  `quantizer -> pre_conv -> pre_transformer -> upsample -> wave decoder`; rerun
  the audio-codec isolation test before returning to generation drift.
- Rerun after the decoder order fix still failed, but with a different scale:
  Rust `peak=1.0`, `rms=0.14575405`, `clip_fraction=2.4346005e-5`;
  Python `peak=0.8203125`, `rms=0.09017588`, `clip_fraction=0.0`;
  waveform preview `max_abs=0.1787872`. This confirms the order fix is
  necessary but insufficient. Continue with per-stage audio-codec activation
  alignment instead of relying on final waveform stats.
- Additional `/tmp/qwen3-tts-rs` structural diffs found and fixed locally:
  - `ConvNeXtBlock` uses GELU after `pwconv1`; local code used SiLU.
  - Vocoder `decoder.0` and final output conv are `CausalConv1d`; local
    `Conv1dConfig` had no left padding. Local initialization now applies
    explicit left padding of 6 for those kernel-7 convs.
  - Wave decoder `CausalConvTranspose1d` trims `kernel_size - stride` samples
    from both left and right; local transposed conv returned the raw output.
    Local wrapper now performs the trim.
  These are all structural, not precision-only, and must be validated with the
  audio-codec isolation test before touching talker generation again.
- Validation after these decoder fixes:
  `cargo test --release -p tts_rs_qwen_burn --test alignment_e2e rust_audio_codec_decodes_python_eager_codes_without_clipping -- --ignored --nocapture`
  passed in 70.16s. This closes the isolated audio-codec white-noise/clipping
  root cause for Python eager codec groups. Remaining bad audio should now be
  investigated in generated codec groups/talker/code-predictor, not in the
  decoder final waveform path unless new evidence appears.
- Full E2E after decoder fixes still fails, but the failure moved back to
  generation:
  `cargo test --release -p tts_rs_qwen_burn --test alignment_e2e e2e_matches_python_oracle -- --ignored --nocapture`
  failed with base talker ids matching Python for the checked prefix, first
  codec mismatch at `step 1 group 6` (`rust=1579`, `python=1217`), and waveform
  preview `max_abs=0.00087161025`. First frame matched Python exactly in this
  diagnostic path. This confirms decoder parity is no longer the active E2E
  blocker for short Python-code previews.
- Default release test after adding transposed-conv trimming initially failed
  only in `audio_codec::tests::wave_decoder_upsample_stage_increases_time`.
  The test still expected raw ConvTranspose length `12` for input length 2,
  kernel 8, stride 4. Reference `CausalConvTranspose1d` trims
  `kernel_size - stride = 4` from both sides, so the correct length is `4`.
  The test expectation was updated to document the reference trimming behavior.
- `cargo test --release -p tts_rs_qwen_burn` then passed. Default test warnings
  remain non-blocking (`SoX` missing, flash-attn missing, tokenizer regex
  warning, and existing unused warnings). Ignored heavy alignment tests are
  still skipped by default.
- CLI regenerated `./0000.wav` after decoder fixes for text
  `你好，欢迎使用语音合成。`: duration `3.496875s`, peak `0.7530093`,
  rms `0.08653868`, `clip_fraction=0.0`.
- Do not compare this CLI manifest's first frame against the default full E2E
  oracle first frame: `py/generate_reference_v9_e2e.py` defaults to
  `其实我真的有发现，我是一个特别善于观察别人情绪的人。`, so the two runs use
  different text. The earlier collect-vs-noncollect suspicion from that
  comparison is rejected as an invalid probe unless reproduced with identical
  text and inputs.

The remaining V9 drift is not proven harmless. It changes generated codec
groups and waveform previews in the ignored E2E oracle test. Treat it as a real
alignment bug until both audio-codec decoding and generated codec groups are
stable.

Diagnostic threshold policy:

- The activation diagnostic tests now use `REPORT_TOLERANCE = 1e-3` instead of
  `5e-2`.
- Rationale: `5e-2` was useful for coarse BF16 screening, but it hid the early
  small drift that later flips marginal codec logits. New probes should use the
  lower threshold to expose the first divergence, then interpret larger BF16
  ULP-sized values in context rather than treating `5e-2` as acceptable.

Current low-threshold code predictor state:

- Production code-predictor attention is back on
  `CastSoftmaxToModelDTypeBeforeValueMatmul`.
- The experimental `PyTorchEagerBf16ScoresAndValueMatmul` remains useful only
  as a targeted probe with Python q/cache. Using it in the production
  autoregressive code-predictor path is rejected because it reintroduces a
  step 0 / head 12 generation flip:
  - Command:
    `QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture`.
  - Rejected production result:
    Rust generated head 12 token `1484`; Python eager generated `344`.
  - After reverting production attention to
    `CastSoftmaxToModelDTypeBeforeValueMatmul`, the same command passed.
- Current focused failure after the revert is step 2 / code-predictor head 6:
  - Command:
    `QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture`.
  - Rust generated:
    `[212, 506, 245, 990, 914, 1543, 310, 1434, 1440, 1673, 743, 1615, 783, 1529, 85, 803]`.
  - Python eager generated:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
  - First mismatching head 6 top-k:
    Rust BF16 `1434=24.625`, `1163=24.5`, `1814=24.5`;
    Python eager `1163=24.75`, `1434=24.5`, `1814=24.375`.
  - Rust F32 lm-head probe still ranks the Rust token first:
    `1434=24.59574`, `1814=24.502563`, `1163=24.461683`.
- Diagnostic cache snapshots added to Rust code-predictor step diagnostics:
  - At step2/head6, layer0 cache is close to Python
    (`key max_abs=0.0078125`, `value max_abs=0.001953125`).
  - Layer1 cache is already broadly shifted
    (`key max_abs=0.48828125`, `value max_abs=0.088134766`), and layers 2-4
    continue accumulating drift.
  - This moves the active root earlier than head6's local attention primitive:
    head6/layer0 can reconstruct Python attention exactly when fed Python
    q/cache, but Rust's layer1+ cache has already been biased by earlier
    heads.
- Rejected narrow attention experiment:
  - `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=8` was used to force
    PyTorch BF16 score emulation only for the head6 cache length.
  - Step2/head6 still failed with Rust choosing `1434` and Python choosing
    `1163`.
  - It reduced the failing head's F32 lm-head gap slightly
    (`1163` moved from `24.461683` to `24.535315`), but not enough to flip the
    output. Therefore the head6 local score primitive alone is not sufficient.
- Rejected additional key-length selective attention experiments:
  - `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=2` still failed at
    step2/head6.
  - `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=2,8` produced a BF16
    three-way tie around the failing head, but the F32 tie-break still selected
    Rust's `1814` instead of Python's `1163`.
  - `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=2,7,8` still failed and made
    the failing-head ordering worse.
  - These runs reject treating one or two key lengths as the whole fix. The
    cache trajectory entering later layers is already biased before head6.
- Rejected all-BF16-score experiment:
  - Extending `PyTorchEagerBf16ScoresAndValueMatmul` to all BF16 query lengths
    made step0 worse: head12 still flipped (`1484` vs `344`) and the final
    head also flipped (`1177` vs `125`).
  - This rejects the hypothesis that the previous global failure was only
    caused by the two-token prefill falling back to F32 score/value math.
- Layer1 cache-by-position check for step2/head6:
  - Layer0 cache is still close, but layer1 cache starts diverging by cache
    position 3.
  - Layer1 key position drift observed in the default run:
    `pos0 max_abs=0.0009765625`, `pos1 max_abs=0.046875`,
    `pos2 max_abs=0.03125`, `pos3 max_abs=0.48828125`.
  - The largest `pos3` sample was head 4 / dim 123:
    Rust `0.98828125`, Python `0.5`.
  - Current implication: debug the head/layer that writes layer1 cache position
    3, rather than re-running head6 local attention probes that are already
    aligned when fed Python q/cache.
- The RMSNorm `1e-6` BF16 tie-bias remains in production. Lowering it to
  `1e-8` is rejected because it regressed step 0 / head 12.

Current conclusion:

- The RMSNorm-looking `0.015625` diff is one BF16 bin around values near
  `2.39`, caused by an F32-to-BF16 cast boundary. It is not harmless: before
  the `1e-6` tie-bias, the one-bin difference was amplified into generated
  token flips.
- The current unresolved step2/head6 mismatch is not fixed by a global PyTorch
  BF16-score emulation path. That emulation exactly matches local attention
  when fed Python q/cache, but the autoregressive hidden/cache trajectory still
  diverges if it is used globally.
- Continue by isolating which earlier head/layer first creates the layer1
  cache drift seen by step2/head6. Do not revisit standalone RMSNorm formula
  swaps without a new failing local RMSNorm probe.

Last default validation:

- `cargo test --release -p tts_rs_qwen_burn` passed after adding the talker
  decode and activation-enhanced code predictor diagnostic tests. Ignored heavy
  tests remain skipped by default.
- `alignment_talker_prefill --ignored` passed as a diagnostic test; it prints
  drift summaries and only fails when
  `QWEN_TTS_STRICT_TALKER_PREFILL_ALIGNMENT=1` is set.
- `alignment_e2e --ignored` still fails, currently with 26 codec mismatches.

## Verified Good Areas

- Frontend/tokenizer/prefill structure aligns with Python in
  `alignment_prefill`.
- The official CustomVoice prefill sequence is used: text embeddings,
  `tts_bos/eos/pad`, codec control tokens, speaker/language IDs, trailing text
  hidden states, and `tts_pad_embed`.
- Talker generation uses EOS and suppresses reserved/control codec tokens.
- Talker decode feeds code-predictor-expanded codec groups back into the next
  talker step.
- Code predictor uses Qwen half-split RoPE instead of Burn even/odd complex
  RoPE.
- Weight import was separately verified by
  `verify_qwen3_tts_talker`: 402 tensors matched exactly.

## Current E2E Failure

Baseline command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e -- --ignored --nocapture
```

Observed before attention FP32 accumulation changes:

- Base talker token ids matched Python exactly.
- Code predictor mismatch started at step 0, group 15:
  `rust=901`, `python=125`.
- Codec mismatches: 44.
- Talker hidden preview `max_abs=0.4375`.
- Waveform preview `max_abs=41.294304`.
- First code predictor step had a marginal top-2 flip:
  `901=23.125`, `125=22.875`.

Observed after FP32 attention score/value matmul experiment:

- First two codec frames matched Python.
- First mismatch moved to step 2, group 6:
  `rust=1636`, `python=1300`.
- Codec mismatches: 31.
- Talker hidden preview `max_abs=0.375`.
- Waveform preview `max_abs=37.64837`.

Observed after combining FP32 attention score/value matmul with RMSNorm
`rsqrt` form:

- First three codec frames matched Python.
- First mismatch moved to step 3, group 4:
  `rust=1296`, `python=610`.
- Codec mismatches: 26.
- Talker hidden preview `max_abs=0.25`.
- Waveform preview `max_abs=41.146408`.

This is an improvement, but still not acceptable as harmless.

## Talker Prefill Activation Probe

Diagnostic oracle:

- Python script: `py/generate_reference_v9_talker_prefill.py`
- Rust test: `tts_rs_qwen_burn/tests/alignment_talker_prefill.rs`
- Default probe captures the first four talker layers.

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_talker_prefill -- --ignored --nocapture
```

Baseline before FP32 attention score/value matmul:

- Earliest `> 5e-2` mismatch appeared at
  `layers.0.mlp.gate`: `max_abs=0.05859375`.
- Layer 0 hidden remained within tolerance:
  `layers.0.hidden.output max_abs=0.0078125`.
- Larger drift accumulated by later layers:
  `layers.1.hidden.output max_abs=0.25`.

Probe result using Python layer-0 MLP input with Rust layer-0 `gate_proj`:

- `probe.layers.0.mlp.gate_from_python_input max_abs=0.0009765625`.
- Conclusion: layer-0 `gate_proj` weights/import and Burn linear with the same
  input are fine; the layer-0 gate drift came from upstream activation drift,
  not from that linear layer.

After FP32 attention score/value matmul:

- Layer 0 no longer has any `> 5e-2` mismatch.
- Layer 0 summary:
  `post_attention_norm max_abs=0.03125`,
  `mlp.up max_abs=0.015625`,
  `mlp.gate max_abs=0.009765625`,
  `hidden max_abs=0.0078125`.
- Remaining larger drift starts at layer 1:
  `layers.1.hidden.output max_abs=0.25`,
  `layers.1.mlp.output max_abs=0.25`.

## Experiments And Results

### FP32 Attention Score/Value Matmul

Change tested:

- Cast `q` and `k` to `F32` before `q @ k^T`.
- Keep softmax in `F32`, then cast weights back to model dtype.
- Cast attention weights and `v` to `F32` for the value matmul, then cast the
  result back to model dtype.

Result:

- Talker prefill layer-0 drift improved substantially.
- E2E codec mismatches improved from 44 to 31 and first mismatch moved from
  step 0 to step 2.
- Keep investigating; this does not prove correctness.

### Attention Scaling Multiply vs Divide

Current status:

- The active code uses `scores * reciprocal(sqrt(head_dim))`.
- This replaced the older division form in the code-predictor eager path.

Result:

- In the current eager-code-predictor branch this fixed a step-1/head-7
  mismatch observed after forcing Python eager attention.
- Older notes that said "do not reapply" were from the previous FP32 attention
  experiment and are obsolete for the current code-predictor eager path.

### RMSNorm `rsqrt` Form

Change tested:

- Replace `x / sqrt(var + eps)` with `x * reciprocal(sqrt(var + eps))`.

Result:

- With current FP32 attention matmul, talker prefill diagnostics were nearly
  unchanged, but E2E improved from 31 mismatches to 26 and moved the first
  mismatch from step 2 to step 3.
- Earlier isolated attempt before the FP32 attention change made E2E worse, so
  keep this only as part of the current combined state and revalidate if
  attention math changes again.

### Residual Add In Model Dtype

Change tested:

- Replace explicit `F32` residual add plus cast with direct tensor add.

Result:

- No visible improvement in the current talker prefill diagnostics.
- Keep under suspicion because PyTorch BF16 addition semantics may differ from
  explicit F32 accumulation, but this experiment alone did not resolve drift.

### SiLU In Model Dtype

Change tested:

- Run `silu(gate)` directly in model dtype instead of `silu(F32(gate))` cast
  back to model dtype.

Result:

- Worse talker prefill diagnostics.
- Do not reapply.

### Python BF16 vs Python F32 Generation

Command used a direct Python comparison of `Qwen3TTSModel.from_pretrained` with
`dtype=torch.bfloat16` and `dtype=torch.float32`.

Result:

- BF16 and F32 Python both matched for the first two codec frames, then diverged.
- Hidden `max_abs` between Python BF16 and F32 reached about `1.93`.
- Conclusion: the generation path is numerically sensitive; even Python dtype
  changes alter later codec groups. This supports strict token/waveform
  validation instead of accepting approximate hidden-state drift as harmless.

## Do Not Repeat Without New Evidence

- Do not call the remaining drift harmless while codec groups differ.
- Do not retry SiLU-in-model-dtype; it worsened diagnostics.
- Do not retry attention reciprocal multiply scaling; it worsened diagnostics.
- Do not retry RMSNorm `rsqrt` as a standalone fix; it only helped after the
  FP32 attention matmul change.
- Do not focus on layer-0 `gate_proj` weight import; probing with Python input
  showed it aligns within `0.0009765625`.

## Next Debug Targets

- Add a code predictor activation oracle for the first mismatching frame and
  group to isolate whether remaining drift is in talker hidden state, code
  predictor cache evolution, or LM head logits.
- Compare Burn/Flex matmul accumulation behavior against PyTorch SDPA on CPU for
  BF16 tensors; Python uses `scaled_dot_product_attention` when the model config
  selects `sdpa`.
- Keep every new probe result in this file before trying another variant.

## Code Predictor Targeted Probe

Added:

- Python script: `py/generate_reference_v9_code_predictor.py`
- Rust test: `tts_rs_qwen_burn/tests/alignment_code_predictor.rs`

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result with Python talker hidden states injected into Rust code predictor:

- Step 0, 1, 3, and 4 matched.
- Step 2 mismatched:
  - Rust: `[212, 506, 245, 990, 914, 698, 1636, 278, 480, 1673, 743, 146, 527, 1759, 85, 1978]`
  - Python: `[212, 506, 245, 990, 914, 698, 1300, 1434, 1728, 1673, 743, 1615, 783, 1956, 85, 803]`
- Step 5 mismatched:
  - Rust: `[462, 506, 245, 990, 1296, 819, 625, 1434, 241, 38, 743, 146, 783, 781, 1772, 1978]`
  - Python: `[462, 506, 245, 990, 1296, 819, 625, 1434, 1440, 1673, 743, 1615, 527, 1838, 1008, 803]`

Conclusion: remaining divergence is not only talker hidden drift. The Rust code
predictor still differs from Python for some Python-hidden inputs. Next probe
should capture code predictor per-head logits/activations around step 2, where
groups 0-5 match and group 6 first diverges.

Additional experiment:

- Reverted attention score matmul back to BF16 for both talker/code predictor
  while keeping the targeted code predictor probe.
- Result was worse: step 0, 1, 2, 3, and 4 mismatched; only step 5 matched the
  autoregressive code predictor output.
- Conclusion: keep FP32 attention score matmul; the remaining code predictor
  drift is not fixed by reverting to BF16 score matmul.

Rejected experiment:

- Recomputing the whole code predictor prefix without cache and without a proper
  causal mask produced many mismatches, including step 0. This is invalid
  because previous positions can attend future tokens in lower layers and then
  pollute later-layer keys/values. Do not use this as an alignment signal.
- Keeping attention softmax weights in F32 for the value matmul, instead of
  casting them back to model dtype before `attn_weights @ V`, made the code
  predictor much worse. Step 1, 3, and 4 mismatched. Keep the BF16 softmax
  weight quantization before value matmul.
- Adding a global BF16 near-tie greedy tolerance of `0.125` made E2E worse:
  first mismatch moved earlier to step 1 group 5 (`rust=63`,
  `python=1412`) and codec mismatches increased to 33. Do not use a global
  tolerance-based greedy policy.

Activation-enhanced probe:

- Added per-head code predictor activation capture to
  `py/generate_reference_v9_code_predictor.py` and
  `tts_rs_qwen_burn/tests/alignment_code_predictor.rs`.
- Command:
  `cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture`.
- Result:
  - The first root mismatch remains `step 2`, predictor `head 5`
    (codec group 6).
  - Heads 0-4 select the same tokens as Python.
  - Head 5 has only small activation/logit drift before selection:
    `logit max_abs=0.34375`; Rust top-2 is
    `[(1636, 26.625), (1300, 26.5)]`, Python top-2 is
    `[(1636, 26.5), (1300, 26.5)]`, while Python generation selects `1300`.
  - Head 6 and later show large activation divergence because the previous
    selected token already differs. Example head 6:
    `layers.4.mlp.product max_abs=9.453125`,
    `model.norm.output max_abs=4.0234375`.
- Step 5 has the same pattern: heads before the first differing group are close,
  and later heads diverge after a marginal selection flip.

Conclusion:

- The code predictor root failure is still a marginal BF16 numerical flip, not
  a structural cache/order bug after the first mismatch.
- Because the selected codec groups and waveform still differ, this cannot be
  accepted as harmless. Continue by reducing the head-5 logits drift or by
  finding the exact PyTorch/Burn arithmetic semantic causing the one-ULP
  top-token change.

Rejected experiment:

- Changed `native_linear_3d` to perform all custom linear matmuls in F32 and
  cast back to the input dtype.
- Code predictor targeted result:
  - The previous `step 2` mismatch disappeared, but a new Python-hidden
    mismatch appeared at `step 4`.
- E2E result:
  - Worse than baseline.
  - First mismatch moved earlier to `step 1 group 6`: `rust=1217`,
    `python=1579`.
  - Codec mismatches increased from 26 to 38.
  - Talker hidden preview worsened to `max_abs=0.4375`.
  - Waveform preview worsened to `max_abs=48.58806`.
- Conclusion: do not use global F32 custom linear accumulation.

Rejected experiment:

- Matched the literal Python eager-attention source more closely by doing the
  value matmul as `BF16 attn_weights @ BF16 value_states` after F32 softmax,
  instead of the current Rust best state (`BF16` softmax weights recast to
  `F32` for the value matmul).
- Targeted code predictor result:
  - Still mismatched at `step 2`, predictor `head 5`; top-2 remained
    `1636` vs `1300`.
- E2E result:
  - Worse than baseline.
  - First mismatch moved earlier to `step 2 group 4`: `rust=243`,
    `python=914`.
  - Codec mismatches increased from 26 to 33.
  - Talker hidden preview worsened to `max_abs=0.40625`.
  - Waveform preview was `max_abs=35.71282`, but token mismatch count and
    earlier divergence make this unacceptable.
- Conclusion: keep the current F32 value matmul even though it is not a literal
  transcription of Python eager attention; it is empirically closer for E2E on
  this backend.

Neutral experiment:

- Changed decoder residual additions to explicit F32 add followed by cast back
  to model dtype.
- E2E result was identical to baseline:
  - Codec mismatches: 26.
  - First mismatch: `step 3 group 4`, `rust=1296`, `python=610`.
  - Talker hidden preview: `max_abs=0.25`.
  - Waveform preview: `max_abs=41.146408`.
- Conclusion: residual-add dtype is not the current differentiator; keep the
  simpler direct tensor add.

Current E2E first mismatch with exact greedy:

- Step 3 group 4: `rust=1296`, `python=610`.
- Rust top-k at that head:
  `[(1296, 26.0), (610, 25.875), (914, 25.625), (243, 25.375), (350, 25.0)]`.
- This supports the theory that small talker-hidden drift flips a marginal code
  predictor decision.

## Talker Decode Activation Probe

Added:

- Python script: `py/generate_reference_v9_talker_decode.py`
- Rust test: `tts_rs_qwen_burn/tests/alignment_talker_decode.rs`

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_talker_decode -- --ignored --nocapture
```

Result:

- The diagnostic test passed structurally and base talker token ids matched.
- Decode steps 0-2 stayed mostly at BF16-level drift:
  - step 0: `codec_head.logits max_abs=0.25`, `model.norm.output max_abs=0.234375`.
  - step 1: `codec_head.logits max_abs=0.25`, `model.norm.output max_abs=0.2890625`.
  - step 2: `codec_head.logits max_abs=0.25`, `model.norm.output max_abs=0.5`.
- Decode step 3 became a large activation divergence:
  - `layers.9.input_norm.output max_abs=1.1738281`.
  - `layers.7.mlp.up max_abs=0.9675293`.
  - `codec_head.logits` was no longer the leading drift, which suggests the
    step input/cache context had already diverged before or at this decode
    step.
- Decode step 4 remained large:
  - `layers.7.mlp.up max_abs=1.4453125`.
  - `codec_head.logits max_abs=0.59375`.

Conclusion:

- Talker decode does not explain the first E2E mismatch by itself. Since base
  talker ids still match, the large step-3 hidden drift is most likely caused
  by Rust feeding different code-predictor-expanded codec groups into later
  talker decode steps.
- Continue with a code predictor activation oracle around the earliest
  Python-hidden mismatch (`step 2`, first differing group `6`) and the current
  E2E mismatch (`step 3`, group `4`).

## Attention Value-Matmul Split

Change:

- Added an explicit attention value-matmul mode so talker and code predictor no
  longer share one global approximation.
- Talker decode/prefill uses the older path:
  `softmax(F32) -> cast weights to model dtype -> value matmul`.
- Code predictor uses the SDPA-closer path:
  `softmax(F32) -> value matmul in F32 -> cast output to model dtype`.

Release checks:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
cargo test --release -p tts_rs_qwen_burn --test alignment_talker_decode -- --ignored --nocapture
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e -- --ignored --nocapture
```

Result:

- Targeted code predictor step `2` passed with Python hidden states.
- Talker decode diagnostic passed structurally and printed the same early small
  BF16-level drift pattern.
- E2E improved compared with the global F32-softmax-retention experiment:
  - `talker hidden preview max_abs=0.25`.
  - `codec mismatches=16`.
  - First mismatch moved to `step 4 group 6`: `rust=2025`,
    `python=243`.
  - The first mismatching top-k was marginal:
    `[(2025, 26.5), (243, 26.375), (625, 26.25), ...]`.

Conclusion:

- The split is directionally correct and should be kept while debugging.
- The remaining failure is still not harmless because codec groups and waveform
  preview differ.

## Talker Decode Step-0 Operator Probes

Added step-0/layer-0 Rust probes to
`tts_rs_qwen_burn/tests/alignment_talker_decode.rs` that recompute downstream
operators from Python intermediate tensors.

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_talker_decode -- --ignored --nocapture
```

Probe result:

- `probe.layers.0.attn_residual_from_python_inputs_and_attn max_abs=0`.
- `probe.layers.0.post_attention_norm_from_python_attn_residual max_abs=0`.
- `probe.layers.0.mlp.gate_from_python_post_norm max_abs=0.000061035156`.
- `probe.layers.0.mlp.up_from_python_post_norm max_abs=0`.
- `probe.layers.0.hidden_from_python_attn_residual_and_mlp_output max_abs=0`.

Conclusion:

- Layer-0 residual adds are not the current differentiator.
- Layer-0 post-attention RMSNorm is not wrong when fed Python
  `attn_residual`.
- Layer-0 MLP `gate_proj` and `up_proj` are not wrong when fed Python
  `post_attention_norm`.
- The remaining step-0 talker drift starts before these downstream operations,
  so continue around talker attention output/input-norm/cache arithmetic.

## Code Predictor Full-Step Recheck After Attention Split

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- The full default step set still failed.
- First reported failure was step 1:
  - Rust selected `[215, 506, 245, 990, 914, 1543, 915, 145, 993, 1004, 743, 903, 527, 1484, 812, 803]`.
  - Python selected `[215, 1782, 375, 657, 145, 1412, 1579, 595, 1611, 1673, 1657, 146, 729, 1759, 1213, 1435]`.
  - Head 0 is the root within that frame: Rust top-k had `506` and `1782`
    tied at `19.0`, while Python top-k had `1782=19.0` and `506=18.875`.
  - Head 1 and later had very large activation differences because the
    previous selected code-predictor token already differed.

Conclusion:

- Treat large later-head drift in this run as autoregressive contamination, not
  as a fresh layer-0 structural bug.
- The real step-1 code predictor root is a marginal head-0 logit ordering
  difference. Continue with teacher-forced or per-head probes that inject the
  Python previous codec token, otherwise every later head hides the first
  numerical flip.

Single-step attention implementation check:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
QWEN_TTS_CODE_PREDICTOR_STEPS=1 QWEN_TTS_CODE_PREDICTOR_ATTENTION=eager cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Default oracle:
  - First mismatch is step 1, head 0.
  - Rust top-k: `[(506, 19.0), (1782, 19.0), ...]`.
  - Python top-k: `[(1782, 19.0), (506, 18.875), ...]`.
  - Head 1 and later are contaminated by Rust selecting `506` instead of
    Python `1782`.
- Eager oracle:
  - Heads 0-7 matched Python selections.
  - First mismatch moved to head 8:
    Rust selected `632`, Python selected `488`.
  - Head 8 top-k was marginal:
    Rust `632=25.0`, `488=24.875`; Python `488=24.75`, `632=24.75`.

Conclusion:

- The first root still depends strongly on attention implementation semantics.
- Rust is closer to eager for the first part of step 1, but production default
  SDPA alignment remains unresolved.
- Do not treat the huge post-head mismatch activations as separate evidence
  until teacher-forced per-head input is used.

Teacher-forced per-head probe:

- Added `teacher_forced_scores` to
  `py/generate_reference_v9_code_predictor.py`.
- Important correction: official Python `code_predictor.forward` applies only
  `lm_head[generation_steps]` to its output, so full-prefix teacher forcing is
  invalid. The oracle now manually loops with `use_cache=True` and forces the
  Python reference token at each head.
- Also corrected `score_topk()` to report the last sequence position for
  `[batch, seq, vocab]` tensors.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- With Python tokens forced back into Rust at every head, the large head1+
  activation/logit divergence collapses back to small numerical drift.
- Teacher-forced head0 still shows the root flip:
  - Rust top-k: `[(506, 19.0), (1782, 19.0), ...]`.
  - Python top-k: `[(1782, 19.0), (506, 18.875), ...]`.
- Teacher-forced heads 1-14 keep the same leading token as Python in this
  step. Representative max logit drift:
  - head1 `max_abs=0.1875`
  - head8 `max_abs=0.3125`
  - head11 `max_abs=0.4375`
  - head14 `max_abs=0.25`

Conclusion:

- The current step1 code-predictor failure is a genuine root flip at head0.
- The large head1+ generated-mode drift is downstream contamination from
  selecting the wrong head0 token.
- Continue by reducing head0 numerical drift, especially through code predictor
  attention and final hidden/logit calculation.

## Code Predictor Attention Kernel Probe: Invalid First Run

Command:

```bash
uv run python py/probe_v9_code_predictor_attention_kernel.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output target/tmp/probe_v9_code_predictor_attention_kernel_step1_head0_layer0.json --step 1 --head 0 --layer 0
```

Initial result:

- The probe reported very large differences between PyTorch SDPA and all manual
  variants for step 1 / head 0 / layer 0 (`pre_o max_abs` about `1.02`,
  `post_o max_abs` about `0.535`).
- This result is invalid as an alignment signal. The captured Python SDPA call
  had `is_causal=true`, while the manual variants in the probe script did not
  apply a causal mask when `attention_mask` was `None`.

Action:

- Updated `py/probe_v9_code_predictor_attention_kernel.py` so manual variants
  apply an upper-triangular causal mask when SDPA uses `is_causal=true`.
- Re-run this probe before drawing conclusions about attention value modes.

Re-run result after fixing the causal mask:

- Step 1 / head 0 / layer 0 now shows only BF16-scale differences between SDPA
  and manual variants.
- `pre_o` comparison against SDPA:
  - `eager`: `max_abs=0.00390625`, `exceed_1e_3=63`.
  - `rust_current`: `max_abs=0.00390625`, `exceed_1e_3=35`.
  - `f32_all_then_cast`: `max_abs=0.001953125`, `exceed_1e_3=9`.
  - `bf16_scores_f32_value`: `max_abs=0.00390625`, `exceed_1e_3=59`.
- `post_o` comparison against SDPA:
  - all tested modes had `max_abs=0.00390625`; `f32_all_then_cast` had the
    fewest values over `1e-3`.

Conclusion:

- The corrected probe does not identify layer-0 attention as a structural
  error. The current Rust code predictor attention mode corresponds most
  closely to `f32_all_then_cast` and should be kept for now.
- Continue by checking later code predictor layers for step 1 / head 0, where
  small BF16 differences may accumulate enough to flip the marginal
  `1782`/`506` logits.

## Code Predictor Step1 Head0 Layer0 Operator Probe

Change:

- Added a targeted Rust test probe for step 1 / head 0 / layer 0 in
  `tts_rs_qwen_burn/tests/alignment_code_predictor.rs`.
- The probe feeds Python intermediate tensors into Rust operators:
  `post_attention_layernorm`, `mlp.gate_proj`, and `mlp.up_proj`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- The generated code groups still diverge at step 1 / head 0:
  Rust selects `506`, Python selects `1782`.
- Before the flip, layer-0 differences are:
  - `layers.0.attn.output max_abs=0.00390625`
  - `layers.0.attn_residual.output max_abs=0.00390625`
  - `layers.0.post_attention_norm.output max_abs=0.0625`
- Targeted operator probe with Python inputs:
  - `probe.step1.head0.layer0.post_norm_from_python_attn_residual max_abs=0`
  - `probe.step1.head0.layer0.gate_from_python_post_norm max_abs=0`
  - `probe.step1.head0.layer0.up_from_python_post_norm max_abs=0.0009765625`

Conclusion:

- Rust RMSNorm and the layer-0 MLP projections are not intrinsically wrong for
  the current failing head when fed Python inputs.
- The remaining root enters at layer-0 attention output. The attention output
  error is only one BF16 step, but the following RMSNorm amplifies it enough
  for later layers to flip a marginal logit.
- Continue by isolating why Rust/Burn attention differs from PyTorch SDPA by
  `0.00390625` in this prefill case, or by finding a numerically equivalent
  attention/output-projection path for this backend.

## Code Predictor Attention Variants And Tie-Break Experiments

Added:

- Python script:
  `py/probe_v9_code_predictor_attention_variants.py`.
- It patches the Python code predictor to run selected manual attention
  variants end-to-end for one code-predictor step while using the same default
  Python talker hidden states.

Command:

```bash
uv run python py/probe_v9_code_predictor_attention_variants.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output target/tmp/probe_v9_code_predictor_attention_variants_step1.json --step 1
```

Python result for step 1:

- Default SDPA groups:
  `[215, 1782, 375, 657, 145, 1412, 1579, 595, 1611, 1673, 1657, 146, 729, 1759, 1213, 1435]`.
- Manual `eager` matched SDPA exactly for this step.
- Manual `rust_current` and `f32_all_then_cast` matched through head 4 but
  diverged at head 5.
- Manual `bf16_scores_f32_value` diverged immediately at head 0.

Rust eager-like experiment:

- Added and tested an `EagerModelDTypeScoresAndValueMatmul` attention mode for
  the Rust code predictor: model-dtype score matmul, FP32 softmax, cast weights
  back to model dtype, then model-dtype value matmul.
- Result:
  - It fixed the previous head-0 selection.
  - It then diverged at head 4/5 and teacher-forced head 5 became worse
    (`rust_topk` led with `1217`, Python led with `1579`).
- Conclusion: do not switch the Rust code predictor to eager-like attention;
  the current F32-score path is still closer after teacher forcing.

Tie-break experiment:

- Tried selecting the largest token id on exact greedy ties only inside the
  code predictor.
- Result:
  - It fixed the head-0 exact tie (`506` vs `1782`).
  - It immediately broke head 2, where Rust tied `657` and `750` but Python
    preferred `657`.
- Conclusion: exact-tie policy is not a reliable alignment fix. Keep the
  default lowest-index greedy behavior.

LM-head isolation:

- Extended `logits_from_python_hidden` diagnostics to print top-k.
- With Python `model.norm.output` fed to Rust `lm_head`, the head-0 top-k
  matched Python exactly:
  `[(1782, 19.0), (506, 18.875), ...]`.
- Conclusion: the head-0 flip is not caused by `lm_head` weights or the final
  linear operation when the hidden state is aligned. It is caused by hidden
  drift before `model.norm.output`, starting from the one-BF16-step layer-0
  attention output difference.

Rejected output-projection experiment:

- Changed only the code-predictor F32 attention path to run attention `o_proj`
  with explicit F32 matmul accumulation and cast back to BF16.
- Result:
  - Step 1 still failed at head 0.
  - `layers.0.attn.output` stayed at the same `max_abs=0.00390625`.
  - Teacher-forced logits changed slightly, but not in a useful direction.
- Conclusion: the remaining layer-0 attention output mismatch is not fixed by
  changing only `o_proj` accumulation to F32. Do not reapply this local
  `o_proj` experiment without new evidence.

Rejected F32 lm-head logits experiment:

- Changed only code-predictor `lm_head` logits to use explicit F32 matmul and
  kept F32 logits for greedy selection.
- Result:
  - `QWEN_TTS_CODE_PREDICTOR_STEPS=1` passed. This fixed the previous step-1
    head-0 `506`/`1782` ordering problem.
  - `QWEN_TTS_CODE_PREDICTOR_STEPS=2` failed. Step 2 diverged at head 5:
    Rust selected `1636`, Python selected `1300`.
  - The step-2 Python top-k had `1636` and `1300` both displayed at `26.5`,
    while F32 Rust logits made `1636` clearly larger.
- Conclusion: keeping all code-predictor lm-head logits in F32 overfits the
  step-1 case and breaks a later step. Do not use global F32 lm-head logits.

## Code Predictor Conservative Tie-Break And Step 3 Drift

Added:

- A conservative code-predictor tie-break in `talker/inference.rs`.
- The normal BF16/native logits remain the source of truth. Rust only consults
  explicit F32 `lm_head` logits when the BF16 top logit is an exact tie and the
  best tied F32 candidate beats the BF16-selected candidate by more than
  `0.0625`, half of one BF16 logit quantum around the observed logits.

Verification:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step 1 passed. The previous head-0 Rust BF16 tie between `506` and `1782`
  is resolved to Python's `1782` because F32 logits favor `1782` by about
  `0.080727`, which exceeds the threshold.
- Step 2 passed. The previous head-5 case remains Python-aligned because the
  F32 gap in the wrong direction is below the threshold.

Next failure:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=3 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Expected step 3:
  `[1181, 506, 245, 990, 610, 407, 1300, 812, 241, 1673, 743, 796, 1160, 781, 1213, 1435]`.
- Actual step 3:
  `[1181, 506, 245, 990, 1296, 1543, 243, 1434, 241, 1144, 743, 362, 706, 1484, 1213, 803]`.
- First generated mismatch is group/head 4, referred to in the current test
  diagnostics as `head 3` because the per-head diagnostic index is zero-based
  after the base-code token.
- Rust generated top-k at the first mismatch:
  `[(1296, 26.0), (610, 25.875), (914, 25.75), ...]`.
- Python top-k:
  `[(1296, 25.875), (610, 25.875), (914, 25.625), ...]`.
- Python selected `610`.
- The F32 lm-head probe for this head favored the wrong Rust token:
  `1296=25.95047`, `610=25.922346`, gap about `0.028`, so the conservative
  tie-break correctly does not apply.

Conclusion:

- The step-3 failure is no longer a simple final-logit tie-break issue. A
  one-BF16-step hidden-state drift before `lm_head` turns a Python tie into a
  Rust `1296` lead.
- Continue with a targeted step-3/head-3 layer probe and compare generated
  versus teacher-forced contexts. Heads 0-2 match, so the cache/token context
  entering the first mismatch should be numerically isolated rather than
  treated as token contamination.

## Step 3 Head 3 Layer-0 Operator Probe

Added:

- Targeted Rust probe for `step3.head3.layer0` in
  `tts_rs_qwen_burn/tests/alignment_code_predictor.rs`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=3 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step 3 still fails at the first generated mismatch:
  - Rust generated head 3 top-k:
    `[(1296, 26.0), (610, 25.875), (914, 25.75), ...]`.
  - Python head 3 top-k:
    `[(1296, 25.875), (610, 25.875), (914, 25.625), ...]`.
  - Python selects `610`; Rust selects `1296`.
- The generated-path activation drift for head 3 begins at layer-0 attention:
  - `layers.0.attn.output max_abs=0.00390625`
  - `layers.0.attn_residual.output max_abs=0.00390625`
  - `layers.0.post_attention_norm.output max_abs=0.03125`
- Targeted operator probe with Python tensors:
  - `probe.step3.head3.layer0.post_norm_from_python_attn_residual max_abs=0`
  - `probe.step3.head3.layer0.gate_from_python_post_norm max_abs=0.000061035156`
  - `probe.step3.head3.layer0.up_from_python_post_norm max_abs=0.0000009536743`
- Teacher-forced step-3 logits remain close for every head and preserve the
  expected token sequence. The large generated-path divergences starting at
  head 4 are downstream contamination from the head-3 token mismatch.

Conclusion:

- The same pattern as the earlier step-1/head-0 failure repeats: RMSNorm and
  MLP are not the intrinsic mismatch when Python intermediates are supplied.
  The decisive drift enters at layer-0 attention output.
- The first mismatch should be debugged as attention numerics/cached context,
  not as an embedding, projection, RMSNorm, MLP, or final `lm_head` weight
  mismatch.

## Step 3 Attention Kernel Variant Probe

Commands:

```bash
uv run python py/probe_v9_code_predictor_attention_kernel.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output target/tmp/probe_v9_code_predictor_attention_kernel_step3_head3.json --step 3 --head 3 --layer 0
uv run python py/probe_v9_code_predictor_attention_variants.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output target/tmp/probe_v9_code_predictor_attention_variants_step3.json --step 3
```

Kernel result for step 3 / head 3 / layer 0:

- `query_dtype=torch.bfloat16`, `key_len=5`, `query_len=1`,
  `has_attention_mask=false`, `is_causal=false`.
- Compared to PyTorch SDPA post-`o_proj` output:
  - `eager max_abs=0.01171875`
  - `rust_current max_abs=0.0078125`
  - `f32_all_then_cast max_abs=0.0009765625`
  - `bf16_scores_f32_value max_abs=0.015625`

End-to-end Python variant result for step 3:

- Default SDPA groups:
  `[1181, 506, 245, 990, 610, 407, 1300, 812, 241, 1673, 743, 796, 1160, 781, 1213, 1435]`.
- `rust_current` groups:
  `[1181, 506, 245, 990, 610, 407, 1300, 812, 241, 1673, 743, 796, 1160, 781, 1213, 1435]`.
- `f32_all_then_cast` groups:
  `[1181, 506, 245, 990, 1296, 1543, 243, 1434, 241, 1144, 743, 362, 706, 1484, 1213, 803]`.

Interpretation:

- The local layer-0 post-`o_proj` closeness is not sufficient to predict greedy
  stability. At this marginal step, keeping softmax weights in F32 through the
  value matmul reproduces the Rust failure sequence.
- The Python variant named `rust_current` casts softmax weights back to BF16
  before the value matmul, then runs the value matmul in F32. That matches
  SDPA's generated groups for step 3.
- This contradicts the earlier assumption that code predictor should keep
  F32 softmax weights for the value matmul. The next experiment should switch
  the Rust code predictor from `KeepSoftmaxF32ForValueMatmul` to
  `CastSoftmaxToModelDTypeBeforeValueMatmul` and re-run step 1, step 2, and
  step 3 before accepting it.

## Rejected Global Code-Predictor BF16-Weight Value Matmul

Experiment:

- Switched both code predictor model paths from
  `KeepSoftmaxF32ForValueMatmul` to
  `CastSoftmaxToModelDTypeBeforeValueMatmul`.

Commands:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step 1 passed.
- Step 2 failed at generated head 5:
  - Rust generated groups:
    `[212, 506, 245, 990, 914, 698, 1636, 278, 480, 1673, 743, 146, 527, 1759, 85, 1978]`.
  - Python groups:
    `[212, 506, 245, 990, 914, 698, 1300, 1434, 1728, 1673, 743, 1615, 783, 1956, 85, 803]`.
  - Rust head-5 top-k:
    `[(1636, 26.625), (1300, 26.5), (915, 25.875), ...]`.
  - Python head-5 top-k:
    `[(1636, 26.5), (1300, 26.5), (915, 25.75), ...]`.
- Targeted operator probe for `step2.head5.layer0` still showed Rust RMSNorm
  and MLP are aligned when fed Python tensors:
  - `post_norm_from_python_attn_residual max_abs=0`
  - `gate_from_python_post_norm max_abs=0.001953125`
  - `up_from_python_post_norm max_abs=0`

Conclusion:

- Do not globally switch code predictor attention to
  `CastSoftmaxToModelDTypeBeforeValueMatmul`. It fixes the step-3 pattern but
  regresses step 2.
- The next check is Python step-2 attention variants. If Python SDPA itself is
  stable under one variant but Rust is not, the fix must be more selective than
  a global attention mode switch.

## Step 2 Variants And Near-Tie Greedy Rule

Command:

```bash
uv run python py/probe_v9_code_predictor_attention_variants.py --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --output target/tmp/probe_v9_code_predictor_attention_variants_step2.json --step 2
```

Result:

- Python SDPA step-2 groups:
  `[212, 506, 245, 990, 914, 698, 1300, 1434, 1728, 1673, 743, 1615, 783, 1956, 85, 803]`.
- Python `rust_current` and `f32_all_then_cast` both matched SDPA for step 2.
- Python `bf16_scores_f32_value` diverged at the same first token as the
  rejected global Rust BF16-weight experiment: `1636` instead of `1300`.

Experiment:

- Reverted the Rust code predictor attention mode to
  `KeepSoftmaxF32ForValueMatmul`.
- Extended code-predictor greedy selection to treat candidates within one BF16
  logit quantum (`0.125`) as near ties only when explicit F32 `lm_head` logits
  do not separate the current BF16 top from the runner-up by more than
  `0.0625`.
- If F32 is decisive, use the F32 winner. If F32 is not decisive, choose the
  lowest id among near-tied candidates, matching stable argmax behavior when
  Python quantizes those logits to a tie.

Rejected intermediate version:

- The first near-tie implementation always selected the lowest id when F32 did
  not pick a different candidate.
- It broke step 1 / head 4:
  - Rust top-k:
    `[(1412, 24.25), (63, 24.125), (1105, 24.125), ...]`.
  - F32 probe:
    `1412=24.262213`, `63=24.100473`, `1105=24.07188`.
  - F32 clearly supported the BF16 top `1412`, so lowering to `63` was wrong.
- The rule was tightened to keep the BF16 top when F32 supports it by more
  than `0.0625`.

Verification:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1,2,3 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Passed for steps 1, 2, and 3.
- This specifically covers:
  - Step 1 / head 0 exact BF16 tie where F32 decisively selects Python's
    `1782`.
  - Step 2 / head 5 exact or near tie where F32 is not decisive enough in the
    opposite direction, preserving Python's lower-id `1300`.
  - Step 3 / head 3 one-ULP near tie where F32 is not decisive enough, selecting
    Python's lower-id `610`.

Conclusion:

- Keep the code predictor attention mode as `KeepSoftmaxF32ForValueMatmul`.
- Keep the narrowed near-tie greedy rule, but still run the default 0-5
  alignment before considering this accepted.

## Default 0-5 With Temporary Wide Tie Margins

Experiment:

- The initial narrowed near-tie rule used:
  - exact BF16 tie F32 margin: `0.0625`
  - one-ULP near-tie F32 margin: `0.05`
  - BF16 near-tie margin: `0.125`

Results:

- A unified `0.05` F32 margin was rejected:
  - It fixed step 5 / head 7, where F32 favored Python's `1440` over `241`
    by about `0.056`.
  - It broke step 2 / head 5, where F32 favored the wrong `1636` over
    Python's lower-id `1300` by about `0.0523` in an exact BF16 tie.
- Splitting exact ties and near ties passed the default code-predictor
  alignment:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Passed for default steps 0-5.

User direction:

- The user rejected wide decision thresholds and required thresholds to be
  `1e-3`.

Conclusion:

- Treat this pass as a diagnostic only, not an accepted final alignment fix.
- Next step: reduce the code-predictor tie-break confidence margins to `1e-3`
  and continue debugging the exposed mismatches.

## Strict 1e-3 Tie-Break Threshold Failure

User direction:

- All decision/report thresholds should be `1e-3`.

Change:

- Set code-predictor F32 exact-tie and near-tie margins to `1e-3`.
- At this point the BF16 near-tie window was still one BF16 quantum; it is
  being reduced after this run as well so no wide threshold remains.

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Failed at step 2 / head 5:
  - Rust generated groups:
    `[212, 506, 245, 990, 914, 698, 1636, 278, 480, 1673, 743, 146, 527, 1759, 85, 1978]`.
  - Python groups:
    `[212, 506, 245, 990, 914, 698, 1300, 1434, 1728, 1673, 743, 1615, 783, 1956, 85, 803]`.
  - Rust BF16 top-k at the first mismatch:
    `[(1300, 26.5), (1636, 26.5), (915, 25.75), ...]`.
  - Python top-k display:
    `[(1636, 26.5), (1300, 26.5), (915, 25.75), ...]`.
  - Python generation selects `1300`, consistent with lower-id argmax on an
    exact BF16 tie.
  - F32 probe:
    `[(1636, 26.507639), (1300, 26.45535), ...]`.

Conclusion:

- With strict `1e-3`, F32 tie-break is not a valid general fix: it changes a
  true BF16 tie into the wrong token at step 2 / head 5.
- The remaining alignment work must reduce the hidden/logit drift that creates
  false Rust ties or one-ULP near ties, rather than deciding them with wide
  heuristic thresholds.

## Strict BF16 Greedy Exposes Step 1 False Tie

Change:

- Removed F32 and near-tie selection from code-predictor greedy.
- Selection now follows BF16 logits directly, with ordinary lowest-index
  behavior for exact ties.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Failed at step 1 / head 0:
  - Rust generated starts with `506`.
  - Python generated starts with `1782`.
  - Rust BF16 top-k:
    `[(506, 19.0), (1782, 19.0), (957, 17.875), ...]`.
  - Python top-k:
    `[(1782, 19.0), (506, 18.875), (957, 17.875), ...]`.
  - F32 lm-head probe over Rust hidden:
    `1782=19.021523`, `506=18.940796`.
- The already-added targeted operator probe still shows:
  - `post_norm_from_python_attn_residual max_abs=0`
  - `gate_from_python_post_norm max_abs=0`
  - `up_from_python_post_norm max_abs=0.0009765625`

Conclusion:

- This is the desired strict failure: step 1 / head 0 is a false Rust BF16 tie,
  caused by hidden drift before `lm_head`, not by final `lm_head` weights.
- Continue debugging layer-0 attention output. The root remains the
  `layers.0.attn.output max_abs=0.00390625` difference that RMSNorm amplifies
  into a logit tie/flip.

## Step 1 Attention Mask And First-Token Probe

Added:

- A targeted probe for step 1 / head 0 / layer 0:
  `probe.step1.head0.layer0.o_proj_from_python_v0`.
- The probe uses Python `layers.0.v_proj.output` for the first prefill token,
  repeats KV heads into attention heads, feeds that directly into Rust
  `o_proj`, and compares against Python `layers.0.attn.output` for the first
  token.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- First-token `v0 -> o_proj` matches Python:
  - `probe.step1.head0.layer0.o_proj_from_python_v0 max_abs=0.0000038146973`
  - `exceed_count=0` at `REPORT_TOLERANCE=1e-3`

Rejected mask experiment:

- I temporarily inverted `build_attention_mask` because Burn 0.21
  `generate_autoregressive_mask` internally calls `tril_mask`.
- This was wrong. Burn's `tril_mask` returns `true` for positions outside the
  lower triangle, i.e. the positions that should be filled/masked.
- Inverting it caused the final prefill query to be fully masked and produced
  NaN logits.

Correction:

- Reverted the causal mask inversion. The original causal mask direction was
  correct.
- The previous `layers.0.attn.output max_abs` index `1736` is in the second
  token (`1024 + 712`) because code predictor hidden size is `1024`; it was
  not a first-token difference.

Conclusion:

- The first-token self-attention path is aligned. The remaining step-1/head-0
  layer-0 attention drift is in the second query, which attends over both
  prefill keys. Continue by probing the second-query attention weights/pre-`o_proj`
  computation from Python q/k/v.

## 2026-05-27 Eager Attention Follow-up

Change:

- Forced Rust code predictor attention calls from
  `KeepSoftmaxF32ForValueMatmul` to
  `EagerModelDTypeScoresAndValueMatmul`.
- Changed the code predictor alignment test default Python attention
  implementation from `default` to `eager`.
- Changed `py/generate_reference_v9_code_predictor.py` argparse default to
  `eager`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Still failed at strict `REPORT_TOLERANCE=1e-3`.
- The first token mismatch moved from step 1 / head 0 to step 1 / head 7:
  - Rust generated:
    `[215, 1782, 375, 657, 145, 1412, 915, 1034, 632, ...]`
  - Python eager generated:
    `[215, 1782, 375, 657, 145, 1412, 915, 1034, 488, ...]`

Conclusion:

- Eager attention is the correct direction and removes the previous head0
  false tie.
- The remaining divergence is now in incremental code-predictor generation
  around head7/cache/position/`generation_steps`, not PyTorch SDPA.
- Next debug target: compare generated vs teacher-forced code-predictor paths
  for step 1 / head 7, especially cache length, input ids/embeds,
  `generation_steps`, and position IDs.

## Step 1 Head7 Root Cause: Attention Scale Operation

Change:

- Changed shared attention scaling from division by `sqrt(head_dim)` to
  multiplication by `1 / sqrt(head_dim)`.
- This matches the Python eager implementation:
  `torch.matmul(query, key.transpose(...)) * module.scaling`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Passed at strict `REPORT_TOLERANCE=1e-3`.

Conclusion:

- The step 1 / head 7 mismatch was not caused by cache length, position IDs,
  or `generation_steps`.
- The root was BF16 eager attention scaling: Burn `div_scalar(sqrt(head_dim))`
  and PyTorch eager `mul(module.scaling)` round differently enough to flip a
  later codebook near-tie.
- Keep the multiplication form in Rust attention so the eager path matches the
  Python oracle.

## Default Code Predictor Run After Eager Scale Fix

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Failed at step 0.
- First generated token mismatch moved to step 0 / head 12
  (codec group index 13):
  - Rust generated:
    `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 1484, 1772, 1177]`
  - Python eager generated:
    `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125]`

Conclusion:

- Step 1 is aligned after the eager scaling fix, but step 0 still has an
  independent late-codebook flip.
- Next debug target: isolate step 0 / head 12, starting from the first
  activation that exceeds `1e-3`.

Correction from the isolated step-0 rerun:

- The previous "head 13" wording was using codec group index language.
- Code predictor head numbering is zero-based after the base token; the first
  mismatch is `head 12`, where Rust chooses `1484` and Python eager chooses
  `344`.

## Rejected Experiment: BF16 SiLU In MLP

Hypothesis:

- Python eager MLP source is `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`.
- Since PyTorch reports BF16 output for `silu(BF16)`, try removing Rust's
  explicit F32 SiLU cast.

Commands:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
QWEN_TTS_CODE_PREDICTOR_STEPS=1 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step 0 passed.
- Step 1 regressed badly:
  - Rust generated:
    `[215, 1782, 375, 657, 145, 63, 1636, 812, 1857, 86, 743, 823, 1108, 2027, 82, 901]`
  - Python eager generated:
    `[215, 1782, 375, 657, 145, 1412, 1217, 145, 241, 1673, 743, 796, 706, 2027, 1213, 803]`
  - First mismatch moved to step 1 / head 4.

Conclusion:

- Direct BF16 `silu` is not a valid global fix. Burn's BF16 SiLU behavior does
  not match the Python eager path closely enough.
- Reverted to the previous F32 SiLU then cast-to-model-dtype implementation.
- Step 0 remains unresolved; continue debugging without this change.

## Rejected Experiment: RMSNorm `powf_scalar(-0.5)`

Finding before the experiment:

- Added targeted layer0 probes for step 0 / head 0 and step 0 / head 12.
- Step 0 / head 12 local layer0 operators align when fed Python
  intermediates:
  - `post_norm_from_python_attn_residual max_abs=0`
  - `gate_from_python_post_norm max_abs=0`
  - `up_from_python_post_norm max_abs=0`
- Step 0 / head 0 still differs when Rust RMSNorm is fed Python
  `attn_residual`:
  - `post_norm_from_python_attn_residual max_abs=0.015625`
  - `exceed_count=7`

Hypothesis:

- Python RMSNorm uses `torch.rsqrt(variance + eps)`.
- Rust used `(variance + eps).sqrt().recip()`.
- Tried `(variance + eps).powf_scalar(-0.5)` as a closer rsqrt-like form.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Still failed at step 0 / head 12.
- Rust still chose `1484`; Python eager still chose `344`.
- The targeted step 0 / head 0 RMSNorm probe remained at
  `max_abs=0.015625`.

Conclusion:

- `powf_scalar(-0.5)` is not the missing Python `rsqrt` behavior.
- Reverted to the previous `sqrt().recip()` implementation.
- The unresolved step0 drift is still introduced before or inside early
  layer0/head0 normalization/attention residual handling.

## Rejected Experiment: Explicit RMSNorm Sum/Divide

Hypothesis:

- Burn `mean_dim` reduction may not match PyTorch `mean`.
- Tried computing RMSNorm variance as
  `x.square().sum_dim(last_dim).div_scalar(hidden_size)`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- No behavioral improvement:
  - Rust still chose `1484` at step 0 / head 12.
  - Python eager still chose `344`.
  - `probe.step0.head0.layer0.post_norm_from_python_attn_residual`
    remained `max_abs=0.015625`.

Conclusion:

- Burn `mean_dim` is not the cause of the observed step0 RMSNorm discrepancy.
- Reverted to `mean_dim`.

## Rejected Experiment: Burn Native RmsNorm

Hypothesis:

- Burn's built-in `RmsNorm::forward` might match the backend's native reduction
  and cast behavior more closely than the Qwen custom wrapper.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step 0 still failed.
- Native `RmsNorm::forward` changed the failure shape:
  - Rust generated:
    `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 1177]`
  - Python eager generated:
    `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125]`
- It fixed the earlier step 0 / head 12 flip, but the final codec group still
  flipped and many head logit `max_abs` values became much larger, including
  head 12 `max_abs=2` and several layer-4 activation drifts above `3`.
- The targeted step 0 / head 0 RMSNorm probe remained:
  `post_norm_from_python_attn_residual max_abs=0.015625`.

Conclusion:

- Burn native RMSNorm is not a valid fix. It can move a marginal argmax, but it
  increases broader activation/logit drift.
- Reverted to the custom Qwen RMSNorm formula:
  cast input to F32, compute mean-square variance, multiply by reciprocal
  sqrt, then cast back to model dtype and multiply by gamma.

## 2026-05-27 Current Rerun: Step 2 / Head 6 Still Fails

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Current first generated mismatch is still step 2 / code-predictor head 6
  (codec group index 7).
- Rust generated:
  `[212, 506, 245, 990, 914, 1543, 310, 1434, 1440, 1673, 743, 1615, 783, 1529, 85, 803]`.
- Python eager generated:
  `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
- First mismatching head 6 top-k:
  - Rust BF16: `1434=24.625`, `1163=24.5`, `1814=24.5`,
    `1034=24.0`, `1345=24.0`.
  - Python eager BF16 display: `1163=24.75`, `1434=24.5`,
    `1814=24.375`, `1345=24.0`, `1034=23.875`.
- Rust F32 lm-head probe for this head still ranks the Rust token first, so
  this is not an exact BF16 argmax tie:
  `1434=24.59574`, `1814=24.502563`, `1163=24.461683`.

Latest attention-cache probe using Python q/cache for step 2 / head 6 / layer 0:

- `eager_bf16_scores_bf16_value max_abs=0.015625`, `exceed_count=551`,
  sample `rust=-1.890625`, `python=-1.90625`.
- `f32_scores_cast_softmax_f32_value max_abs=0.024902344`,
  `exceed_count=684`, sample `rust=-0.036376953`,
  `python=-0.011474609`.
- `f32_scores_f32_softmax_f32_value max_abs=0.025146484`,
  `exceed_count=692`, sample `rust=-0.036621094`,
  `python=-0.011474609`.

Conclusion:

- The current failure is real and cumulative: by step 2 / head 6 the Rust
  hidden state before `lm_head` is already biased toward `1434`.
- The layer-0 attention recompute from Python q/cache suggests the BF16
  eager-like score/value path is locally closer to Python for this failing
  case than the current F32 score/value variants, but older step-0/step-1
  experiments showed a global switch alone is insufficient. Continue by
  isolating whether the first layer-0 difference is attention weights,
  pre-`o_proj` value matmul, or `o_proj`.

## Step 2 / Head 6 Attention Weights Root

Change:

- Extended `probe_code_predictor_attention_from_python_cache` to compare
  attention weights directly and to rebuild attention output from Python
  weights plus Python value cache.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result for `step2.head6.layer0`:

- Weight comparison from Python q/cache:
  - `eager_bf16_scores max_abs=0.01171875`, `exceed_count=3`.
  - `f32_scores max_abs=0.01953125`, `exceed_count=18`.
- Output comparison:
  - `eager_bf16_scores_bf16_value max_abs=0.015625`,
    `exceed_count=551`.
  - `f32_scores_cast_softmax_f32_value max_abs=0.024902344`,
    `exceed_count=684`.
  - `f32_scores_f32_softmax_f32_value max_abs=0.025146484`,
    `exceed_count=692`.
  - `python_weights_bf16_value max_abs=0`, `exceed_count=0`.
  - `python_weights_f32_value max_abs=0`, `exceed_count=0`.

Conclusion:

- The step2/head6 layer-0 mismatch is definitely before value matmul and
  `o_proj`: Python weights plus Python value cache reconstruct the captured
  Python attention output exactly in Rust.
- The first differing operation for this case is attention score/softmax:
  `q @ k^T * scaling -> softmax`. The BF16 eager score path is closer than the
  F32 score path, but still not exact. Continue by capturing or reconstructing
  Python pre-softmax scores to determine whether the difference is in the dot
  product accumulation, scale rounding, mask, or softmax cast boundary.

## Rejected Experiment: Global Code Predictor BF16 Eager Attention

Hypothesis:

- Since the `step2.head6.layer0` Python-cache probe showed BF16 eager attention
  weights closer than F32 score variants, switch both code-predictor forward
  paths to `EagerModelDTypeScoresAndValueMatmul`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Still failed at step 2 / head 6.
- Rust still chose `1434`; Python eager chose `1163`.
- Rust top-k at the failing head became:
  `1434=24.75`, `1163=24.625`, `1814=24.625`, `278=24.125`,
  `1034=24.125`.
- F32 lm-head probe still favored the Rust token:
  `1434=24.728907`, `1814=24.655243`, `1163=24.574837`.

Conclusion:

- A global switch to BF16 eager attention does not fix the cumulative hidden
  drift. It is locally closer for one layer-0 attention-weight probe, but the
  full decoder stack still ends up biased toward the Rust token.
- Reverted code-predictor attention to
  `CastSoftmaxToModelDTypeBeforeValueMatmul` while continuing to isolate the
  score/softmax boundary.

## Step 2 / Head 6 Pre-Softmax Score Capture

Change:

- Python oracle now captures `layers.N.attn.scores` and
  `layers.N.attn.manual_weights` by reconstructing the official eager path:
  `torch.matmul(query, key.transpose(2, 3)) * scaling`, optional mask add,
  then `softmax(..., dtype=torch.float32).to(query.dtype)`.
- Rust probe compares BF16-score and F32-score variants directly against the
  captured Python scores.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result for `step2.head6.layer0`:

- `python_captured_weights_vs_manual_weights max_abs=0`, `exceed_count=0`.
  This confirms the Python manual score/softmax reconstruction exactly
  matches the attention weights returned by the official eager implementation.
- Rust scores from Python q/cache:
  - `eager_bf16_scores max_abs=0.0625`, `exceed_count=1`,
    sample `rust=11.8125`, `python=11.875`.
  - `f32_scores max_abs=0.050320625`, `exceed_count=98`,
    sample `rust=11.824679`, `python=11.875`.
- The downstream weight/output comparisons remain:
  - `eager_bf16_scores` weights: `max_abs=0.01171875`.
  - `f32_scores` weights: `max_abs=0.01953125`.
  - Python weights + Python value cache: output `max_abs=0`.

Conclusion:

- The first confirmed differing primitive for the current failure is the BF16
  attention score dot product itself, not mask direction, scale placement,
  softmax, value matmul, `o_proj`, or layout.
- Around the decisive score, Python eager's BF16 `torch.matmul` lands on
  `11.875`; Burn's BF16 matmul lands on `11.8125`; Rust F32 accumulation is
  `11.824679`. This is a backend BF16 matmul/rounding boundary, but it is not
  harmless because the resulting softmax weight drift is amplified by RMSNorm,
  MLP, and later decoder layers into a different generated codebook token.
- Next target: test a Rust attention-score path that emulates PyTorch eager
  BF16 score rounding for the small autoregressive code-predictor score
  matrices, instead of trying more RMSNorm variants.

## PyTorch BF16 Score Emulation And RMSNorm Bias Check

Change:

- Added a Rust code-predictor attention score mode that emulates the observed
  Python eager BF16 score behavior:
  F32 dot-product sum, BF16 round the unscaled sum, multiply by scaling, then
  BF16 round the scaled score.
- Extended the head6 attention-cache probe across layers 0-4.

Validation probe:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- With Python q/cache, the new `pytorch_bf16_scores` probe exactly matches
  Python scores, weights, and attention outputs for step2/head6 layers 0-4.
- Example layer0:
  - `attention_scores_from_python_cache.pytorch_bf16_scores max_abs=0`.
  - `attention_weights_from_python_cache.pytorch_bf16_scores max_abs=0`.
  - `attention_from_python_cache.pytorch_bf16_scores_bf16_value max_abs=0`.
- Example layer3:
  - previous F32 score path was badly wrong for large scores:
    `f32_scores max_abs=5.211914`.
  - `pytorch_bf16_scores max_abs=0`.

Remaining result:

- Full step2 still fails at head6: Rust still chooses `1434`, Python chooses
  `1163`.
- The local attention primitive is now aligned when fed Python intermediates,
  so the remaining problem is hidden/cache drift entering the layer stack, not
  score/softmax/value/`o_proj` math in isolation.

Rejected RMSNorm experiment:

- Temporarily reduced the production RMSNorm BF16 tie bias from `1e-6` to
  `1e-8`.
- Command:
  `QWEN_TTS_CODE_PREDICTOR_STEPS=0 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture`.
- Result: regressed step0/head12. Rust chose `1484`; Python chose `344`.
- The existing targeted RMSNorm variants explain why:
  - step0/head0 `signed_tie_bias_1e_6 max_abs=0`.
  - step0/head0 `signed_tie_bias_1e_8 max_abs=0.015625`.

Conclusion:

- Keep the RMSNorm tie bias at `1e-6`; lowering it reintroduces the earlier
  step0 RMSNorm boundary flip.
- The PyTorch BF16 score emulation is a valid primitive fix, but it is not
  sufficient by itself. Continue with the earliest remaining hidden/cache drift
  after the score fix.

## Step 2 / Head 2 Writes The Layer1 Pos3 Drift

Change:

- Added step2/head2 local probes because layer1 cache position 3 is written
  while processing the token used to produce code-predictor head 2.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- The overall failure is unchanged: step2/head6 still selects Rust `1434`
  while Python eager selects `1163`.
- The new step2/head2 probes show layer1 operators are not intrinsically wrong
  when fed Python intermediates:
  - `layer1.input_norm_from_python_hidden max_abs=0.000061035156`.
  - `layer1.q_proj_from_python_input_norm max_abs=0`.
  - `layer1.k_proj_from_python_input_norm max_abs=0.00048828125`.
  - `layer1.v_proj_from_python_input_norm max_abs=0`.
  - `layer1.q_norm_from_python_q_proj max_abs=0.000061035156`.
  - `layer1.k_norm_from_python_k_proj max_abs=0.001953125`, one value over
    the `1e-3` reporting threshold.
  - `layer1.hidden_from_python_residual_and_mlp max_abs=0`.
- Step2/head2 generated-path activations still show the large layer1 cache
  source:
  - `layers.1.k_norm.output max_abs=0.48828125`.
  - `layers.1.k_rot.output max_abs=0.48828125`.

Conclusion:

- The large layer1 cache drift is not caused by layer1 RMSNorm/projection/MLP
  code when the input is aligned.
- The drift enters before layer1, through layer0 hidden-state drift. RMSNorm
  then amplifies a small layer0 attention/residual difference because it rescales
  the whole vector by the inverse RMS and BF16-quantizes the result. This is why
  a `0.00390625` layer0 attention difference can later appear as
  `0.48828125` in layer1 key normalization.
- Continue at step2/head2 layer0 attention and the RMSNorm cast boundary. Do
  not treat the large layer1 RMSNorm-looking diff as an independent RMSNorm
  formula bug unless a Python-input local RMSNorm probe fails.

## Rejected Narrow KeyLen 4 Score Emulation

Hypothesis:

- Since layer1 cache position 3 is written by the step2/head2 path, force
  PyTorch BF16 score emulation only at `key_len=4`.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step2/head6 still failed:
  Rust selected `1434`; Python eager selected `1163`.
- The layer1 cache position drift changed shape:
  - Before this experiment, layer1 key pos3 was the largest:
    `pos3 max_abs=0.48828125`.
  - With `key_len=4`, pos3 reduced to `max_abs=0.046875`.
  - The largest layer1 key drift moved to pos4:
    `pos4 max_abs=0.3400879`.
  - Layer1 value pos3 also reduced from `0.088134766` to `0.00390625`, while
    pos4 remained `0.08105469`.

Conclusion:

- `key_len=4` confirms that the pos3 cache drift is caused by the attention
  score path for that write, but fixing only one write position is not enough.
- The next positions keep accumulating the same class of drift. The next probe
  should test a contiguous key-length range instead of another single
  key-length patch.

## Partial Improvement: KeyLen 4 Through 8 Score Emulation

Hypothesis:

- If `key_len=4` only moves the drift to the next cache position, force
  PyTorch BF16 score emulation over the contiguous range that writes positions
  3 through 7.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Step2 still failed, but the first generated mismatch moved later:
  - Rust:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1351, 743, 362, 914, 1484, 85, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
- The earlier head6 mismatch (`1434` vs `1163`) is fixed by this range.
- Layer1 cache drift improved substantially:
  - Layer1 key max dropped to `0.046875`.
  - Layer1 value max dropped to `0.0078125`.
  - Positions 3 through 7 no longer show the previous `0.34` to `0.49`
    key drift.
- The new first mismatch is head8:
  - Rust top: `1351=24.875`, `1004=24.75`.
  - Python top: `1351=24.75`, `1004=24.75`, selecting `1004`.

Conclusion:

- The contiguous score-emulation range confirms the main cache accumulation
  mechanism, but it is still not a complete fix.
- Continue by extending the range to later key lengths, then re-check earlier
  steps because earlier notes showed all-key BF16 score emulation could regress
  step0.

## Strict Greedy And Extended KeyLen Ranges

Change:

- Removed the code-predictor F32 tie-break from generation. Greedy selection now
  uses the same BF16 logits and lowest-index exact-tie behavior as
  `sample_token`.

Reason:

- With `key_len=4..16`, head8 produced an exact BF16 tie between `1004` and
  `1351`.
- Python eager generation selected `1004` by lowest-index argmax semantics.
- Rust's F32 tie-break selected `1351`, creating a false divergence that was
  not a model arithmetic difference.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8,9,10,11,12,13,14,15,16 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result after removing the F32 tie-break:

- Head8 tie reversal disappeared.
- First mismatch moved to head11:
  - Rust:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 783, 1759, 85, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
- The failing head11 is not an exact tie:
  - Rust top: `783=25.375`, `914=25.25`.
  - Python top: `914=24.875`, `783=24.625`.

Rejected range:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

- This regressed back to the old head6 mismatch:
  Rust `1434`; Python `1163`.
- It also changed early logits substantially. Example head0 logit drift became
  `max_abs=0.20019531`.

Conclusion:

- Removing F32 tie-break is a real fix and should stay.
- PyTorch BF16 score emulation is beneficial for the generated one-token
  decode range starting at key_len 4, but applying it to prefill/key_len 2 and
  early key_len 3 is rejected.
- Continue with targeted head11 probes under the `4..16` diagnostic range
  instead of broadening to all key lengths.

## Extended KeyLen Range Follow-up

These results were recorded after removing the F32 code-predictor tie-break.
They supersede earlier notes that still included the F32 tie-break behavior.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=3,4,5,6,7,8,9,10,11,12,13,14,15,16 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Still failed at step 2 / head 11.
- The generated sequence matched the `4..16` range:
  Rust selected `783` where Python selected `914`.

Conclusion:

- Including key_len `3` does not improve the remaining mismatch.
- Keep key_len `3` excluded unless a new local probe shows a direct need for
  that prefill-adjacent cache length.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8,9,10,11,12 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- The previous head11 mismatch was fixed.
- First mismatch moved to head12:
  - Rust:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1759, 82, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.

Conclusion:

- Extending score emulation through key_len `12` is useful but incomplete.
- Adding key_len `13..16` changes the trajectory enough to reintroduce a
  head11 mismatch, so this is not a simple contiguous "more is better" range.

## Step 2 / Head 11 Local Probe Under KeyLen 4..16

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8,9,10,11,12,13,14,15,16 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- The full generated path still failed at step 2 / head 11:
  Rust selected `783`; Python selected `914`.
- Python-input local operators did not expose a structural layer bug:
  - layer0 `post_norm_from_python_attn_residual max_abs=0.001953125`.
  - layer0 gate/up from Python post-norm matched at `max_abs=0`.
  - layer1 through layer4 local hidden recomposition stayed at BF16-scale
    drift, with `hidden_from_python_residual_and_mlp` equal to `0` for the
    probed recompositions except layer4 at `0.0078125`.
- Attention recompute from Python q/cache for head11 layers 0-4 matched under
  `pytorch_bf16_scores`:
  scores, weights, and attention outputs were exact or near-zero against the
  Python eager capture. Layer3 was a decisive check because F32-score variants
  were several units away while `pytorch_bf16_scores` matched.

Conclusion:

- The head11 mismatch is not caused by the isolated attention primitive when
  Python q/cache are supplied.
- The remaining error is cumulative hidden/cache drift before head11. Continue
  by identifying which key_len/layer application of PyTorch BF16 score
  emulation helps or hurts the generated trajectory, instead of changing
  RMSNorm or final logits.

## KeyLen 4 Through 8 Rerun After Strict Greedy

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Still failed at step 2, but the old head6 mismatch was fixed:
  - Rust:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1351, 743, 362, 914, 1484, 85, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
- First mismatch is head8:
  - Rust top-k starts `1351=24.875`, `1004=24.75`.
  - Python top-k displays `1351=24.75`, `1004=24.75` and generation selects
    lower id `1004`.
- This is no longer the old exact-tie-only failure after F32 tie-break removal;
  Rust has a real one-BF16-step lead for `1351`.

Conclusion:

- `key_len=4..8` is a partial fix only. It removes head6 but leaves cumulative
  drift large enough to flip head8.
- Continue testing selective score-emulation schedules; do not reintroduce any
  F32 or wide-threshold tie-break to hide this.

## Layer-Filtered Score Emulation Probe

Change:

- Added diagnostic-only environment variable
  `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER`.
- Existing behavior is preserved when it is unset: key_len matching applies to
  all code-predictor layers.
- When set, PyTorch BF16 score emulation is applied only if both key_len and
  layer index match.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER=0 QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Failed exactly like the default step2/head6 path:
  - Rust:
    `[212, 506, 245, 990, 914, 1543, 310, 1434, 1440, 1673, 743, 1615, 783, 1529, 85, 803]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.

Conclusion:

- Applying PyTorch BF16 score emulation to layer0 alone over key_len `4..8`
  does not fix the cache trajectory.
- Continue testing layer1+ and combinations. Do not conclude from layer0 local
  score probes alone that layer0-only is the production fix.

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER=1 QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Regressed earlier than the default failure:
  - Rust:
    `[212, 506, 245, 990, 610, 785, 915, 1434, 480, 1333, 743, 175, 527, 1484, 1008, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.
- The first mismatch is head3: Rust has a BF16 tie
  `610=25.75`, `914=25.75` and selects lower id `610`; Python has
  `914=25.75`, `610=25.5`.

Conclusion:

- Layer1-only score emulation is harmful. It creates an early false tie and
  contaminates all later heads.
- Any accepted layer schedule must either exclude layer1 or counterbalance it
  with other layers; do not use layer1-only as a fix.

Batch layer-combination search:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER=<layers> QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=4,5,6,7,8 QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Results:

- No tested layer subset passed step 2.
- Single-layer `2`, `3`, and `4` behaved like default and still failed at
  head6:
  Rust `1434`; Python `1163`.
- Most combinations that include layer1 also behaved like default, or
  regressed earlier. Examples:
  - `0,1` failed early at head3:
    Rust `[212, 506, 245, 990, 610, 407, 1300, ...]`.
  - `1,3,4` failed at head6 with Rust `1814` instead of Python `1163`.
- Best family so far is layer0 plus layer3, excluding layer1:
  - `0,3`:
    Rust `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1759, 82, 1978]`.
  - `0,2,3`:
    Rust `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1759, 85, 1978]`.
  - `0,2,3,4`:
    Rust `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1759, 85, 1978]`.
  - Python:
    `[212, 506, 245, 990, 914, 1543, 310, 1163, 1857, 1004, 743, 362, 914, 1529, 85, 1978]`.

Conclusion:

- Layer3 is the first useful positive layer in the generated trajectory, but it
  requires layer0 to have an effect.
- Layer1 is currently a negative layer for this key_len range.
- Continue by varying key_len ranges for `layers=0,3` and related
  layer0+layer3 schedules. The active mismatch has moved from head6 to head13,
  so the next fix should target the later cache writes rather than revisiting
  RMSNorm formula changes.

## Layers 0,3 KeyLen Range Probe

Command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER=0,3 QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=<keys> QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Results:

- `4,5,6,7,8,9` regressed to head8:
  Rust selected `1351`; Python selected `1004`.
- `4,5,6,7,8,9,10` through `4..16` all returned to the later head13
  mismatch:
  Rust selected `1759`; Python selected `1529`.
- Removing key_len `4` is harmful:
  - `5..12` failed early at head3 with Rust `610` instead of Python `914`.
  - `6..12` also failed early at head3 with Rust `610`.

Conclusion:

- key_len `4` is required for the useful `layers=0,3` trajectory.
- key_len `9` alone is harmful, but adding key_len `10` counteracts that
  earlier head8 regression and returns to the head13 failure.
- The schedule is not a monotonic contiguous range. Continue with noncontiguous
  key sets, especially `4..8` plus selected later key lengths while avoiding
  broad threshold tie-breaks.

Noncontiguous follow-up:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_LAYER=0,3 QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_KEY_LEN=<noncontiguous-keys> QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Results:

- Tested `4..8` plus selected later key lengths:
  `10`, `11`, `12`, `13`, `10,11`, `10,12`, `10,13`,
  `10,11,12`, `10,11,12,13`, and `10,11,12,13,14,15,16`.
- None passed step 2.
- All retained the same head13 mismatch:
  Rust selected `1759`; Python selected `1529`.
- Some variants also kept Rust head14 as `82`; others matched Python head14
  `85`, but the first failure remained head13 in all cases.

Conclusion:

- For `layers=0,3`, later key_len additions cannot fix head13 by themselves.
- Need a per-key, per-layer schedule probe. The next diagnostic switch should
  allow schedules like `4-8:0,3;13:2` instead of applying one layer set to all
  selected key lengths.

## End-to-End CLI Audio Probe

User direction:

- Temporarily stop focusing on code-predictor correction.
- First produce end-to-end inference audio and judge whether the output audio
  is correct.

Change:

- Extended `qwen3-tts` CLI `manifest.json` with:
  - request metadata,
  - talker token count and token ids,
  - codec frame preview,
  - raw waveform shape/duration/min/max/peak/RMS/mean/clip fraction,
  - `audio_status.verdict`.

Rust command:

```bash
cargo run --release -p tts_rs_qwen_burn --bin qwen3-tts -- --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --text "你好，欢迎使用语音合成。" --language Chinese --speaker Vivian --output-dir target/tmp/qwen3_tts_cli_probe --max-new-tokens 64
```

Rust result:

- CLI completed and wrote:
  - `target/tmp/qwen3_tts_cli_probe/0000.wav`
  - `target/tmp/qwen3_tts_cli_probe/manifest.json`
- WAV container is valid PCM:
  `RIFF/WAVE, 16-bit mono, 24000 Hz`.
- Manifest waveform stats:
  - shape `[1, 1, 82149]`
  - duration `3.422875s`
  - min `-53.954876`
  - max `65.90345`
  - peak `65.90345`
  - RMS `15.520979`
  - clip_fraction `0.9464875`
  - verdict `invalid_clipped`
- PCM inspection confirms the saved WAV is mostly saturated:
  - `peak_i16=32767`
  - `rms_i16=32178.61`
  - `clipped_fraction=0.946402`
  - first 16 samples are all `32767`.

Python reference command:

```bash
uv run python <inline reference script>  # same text/language/speaker/max_new_tokens
```

Python result:

- Wrote `target/tmp/qwen3_tts_python_probe/0000.wav`.
- Reference stats:
  - frames `82560`
  - duration `3.44s`
  - min `-0.75390625`
  - max `0.8125`
  - peak `0.8125`
  - RMS `0.091216`
  - clip_fraction `0.0`
- Python first codec frame:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125]`.
- Rust first codec frame:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 1177]`.

Conclusion:

- End-to-end Rust audio output is implemented and produces a WAV file, but the
  audio is not correct.
- This is not just a WAV writer scaling issue: Python reference output is in a
  normal `[-1, 1]` range for the same input, while Rust raw waveform is far
  outside that range and clips on save.
- The first codec frame already differs at the final codebook (`1177` vs
  Python `125`), matching the known code-predictor alignment failure. Continue
  treating code-predictor/talker alignment as the root cause before judging
  audio quality again.

## Interrupted Schedule Probe Before Audio Pivot

Change:

- Added diagnostic-only schedule env
  `QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_SCHEDULE`.
- Syntax examples:
  - `4-8:0,3`
  - `4-8:0,3;14:1,2`
- If set, the schedule overrides the older key_len/layer env filters for the
  PyTorch BF16 score-emulation probe.

Partial command:

```bash
QWEN_TTS_CODE_PREDICTOR_PYTORCH_BF16_SCHEDULE="4-8:0,3;14:<layers>" QWEN_TTS_CODE_PREDICTOR_STEPS=2 cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Partial result before the user redirected to end-to-end audio:

- Tested many `14:<layers>` subsets, including single layers, pairs, and most
  triples/quads.
- None fixed the head13 mismatch.
- All observed variants still selected Rust `1759` where Python selected
  `1529`.
- Some variants changed the next head from Rust `82` to Python's `85`, but
  the first mismatch remained head13.

Conclusion:

- Local key_len 14 layer selection is not sufficient to repair the head13
  decision on top of base schedule `4-8:0,3`.
- This probe was intentionally stopped when the user requested prioritizing
  end-to-end audio output and audio correctness judgement.

## 2026-05-27 Same-Text Code Predictor Step0 Recheck

Context:

- User requested continuing debug until alignment and keeping all tolerance
  gates at `1e-3`.
- Rechecked the CLI text that produced the clipped WAV:
  `你好，欢迎使用语音合成。`
- Python oracle is forced to eager attention by
  `py/generate_reference_v9_code_predictor.py`.

Command:

```bash
QWEN_TTS_TEXT="你好，欢迎使用语音合成。" \
QWEN_TTS_CODE_PREDICTOR_STEPS=0 \
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

Result:

- Failed.
- Rust groups:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125]`
- Python groups:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 2027, 1772, 125]`
- First mismatch is head 12: Rust selects `344`, Python selects `2027`.
- Head 12 top candidates are close:
  - Rust: `344=23.0`, `2027=22.875`, `1484=22.625`
  - Python: `2027=22.875`, `344=22.625`, `1860=22.125`

Attention probe result for `step0.head12`:

- With Python cache tensors injected, `pytorch_bf16_scores` matched Python
  captured attention scores/weights exactly for layers 0-4.
- The default Rust F32 score path does not match those captured eager tensors.

Rejected experiment:

```bash
QWEN_TTS_TEXT="你好，欢迎使用语音合成。" \
QWEN_TTS_CODE_PREDICTOR_STEPS=0 \
QWEN_TTS_CODE_PREDICTOR_ATTENTION=pytorch_bf16 \
cargo test --release -p tts_rs_qwen_burn --test alignment_code_predictor -- --ignored --nocapture
```

- Failed.
- Rust groups changed to:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 901]`
- Python groups remained:
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 2027, 1772, 125]`

Conclusion:

- Do not globally switch production code predictor attention to the
  `pytorch_bf16` diagnostic mode. It improves local Python-cache attention
  probes but does not align the full generated trajectory for this input.
- Next debug pass should compare the Rust reference repos' code-predictor
  generation structure, especially cache vs full-sequence recomputation,
  position/cache semantics, and sampling/argmax behavior.

Correction after inspecting the generated oracle JSON:

- `target/tmp/reference_v9_code_predictor.json` currently has
  `full_codec_groups[0]` equal to
  `[1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 2027, 1772, 125]`.
- Therefore the older note that Python's first frame used head12 `344` is stale
  for the current eager oracle state.
- Current Rust default code predictor head12 `344` is a real mismatch against
  the current Python eager full generation, not only a standalone re-generation
  artifact.

## 2026-05-27 Rust Reference Code-Predictor Cache Check

Reference repos inspected:

- `/tmp/qwen3-tts-rs` from `git@github.com:danielclough/qwen3-tts-rs.git`.
- `/tmp/qwen3-burn` from `git@github.com:eidola-ai/qwen3-burn.git`.

Findings:

- `qwen3-tts-rs/qwen_tts/src/nn/code_predictor.rs` contains two paths:
  - `generate`: full-sequence recomputation after each predicted codebook.
  - `generate_with_cache`: cached autoregressive path.
- Its call sites use `generate_with_cache`.
- However `qwen3-tts-rs` passes only `last_hidden` into
  `generate_with_cache`, while the official Python code predictor path passes
  `torch.cat((past_hidden, last_id_hidden), dim=1)` where `last_id_hidden` is
  the base codebook embedding. The official Python path is still the stronger
  semantic reference for input construction.
- `qwen3-burn` attention uses a straightforward Burn-native BF16 matmul,
  softmax, and value matmul path. This is useful as a performance-oriented
  Rust reference, but it does not solve the strict Python eager alignment by
  itself.

Diagnostic experiment:

- Added temporary env `QWEN_TTS_CODE_PREDICTOR_FULL_RECOMPUTE=1`.
- This uses official input construction but recomputes the whole code-predictor
  sequence per head instead of reusing KV cache.

CLI command:

```bash
QWEN_TTS_CODE_PREDICTOR_FULL_RECOMPUTE=1 \
cargo run --release -p tts_rs_qwen_burn --bin qwen3-tts -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64
```

Result:

- First frame improved only at the final codebook:
  - default cached first frame ended with `..., 1772, 1177`
  - full recompute first frame ended with `..., 1772, 125`
- It did not fix head12: first frame still has `344` while current Python eager
  full generation has `2027`.
- Audio is still invalid/clipped:
  - peak `65.727135`
  - RMS `15.721404`
  - clip_fraction `0.94906324`
  - verdict `invalid_clipped`

Conclusion:

- Code-predictor cache semantics contribute to at least the final codebook
  mismatch, but they are not the primary end-to-end audio failure.
- The talker trajectory is already diverging: full recompute changed the next
  talker tokens (`1995, 215, 212, 1181, 462, 462, ...`) while default cached
  had (`1995, 215, 212, 1181, 462, 333, ...`), and neither matches current
  Python reference behavior.
- Keep the full-recompute switch as a diagnostic only; do not promote it to
  production without fixing strict alignment.

## 2026-05-27 Python V9 Eager Enforcement Gap

Finding:

- `py/generate_reference_v9_code_predictor.py` already forced the wrapper,
  talker, and code-predictor configs to `_attn_implementation = "eager"`.
- `py/generate_reference_v9_e2e.py`,
  `py/generate_reference_v9_prefill.py`,
  `py/generate_reference_v9_talker_decode.py`, and
  `py/generate_reference_v9_talker_prefill.py` did not force eager.
- This violated the current rule that Python reference scripts must avoid
  PyTorch SDPA and use eager attention only.

Change:

- Added `force_eager_attention(wrapper)` to those V9 scripts immediately after
  `Qwen3TTSModel.from_pretrained(...)`.

Consequence:

- Older E2E notes generated without explicit eager may not be comparable with
  the current strict code-predictor oracle.
- Re-generate all V9 Python references before using them for new conclusions.

## 2026-05-27 Model Isolation: Audio Codec Is Independently Clipping

User guidance:

- Split model tests to reduce scope.
- Check Rust reference implementations first to ensure the functional structure
  is correct, then use Python eager as numerical oracle.

Change:

- Added ignored test
  `rust_audio_codec_decodes_python_eager_codes_without_clipping` in
  `tts_rs_qwen_burn/tests/alignment_e2e.rs`.
- The test generates Python eager codec groups for
  `你好，欢迎使用语音合成。` and feeds those groups directly into the Rust
  audio codec decoder. This bypasses Rust frontend, talker, and code predictor.

Command:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e \
  rust_audio_codec_decodes_python_eager_codes_without_clipping -- --ignored --nocapture
```

Result:

- Failed.
- Rust audio codec stats when decoding Python eager codec groups:
  - min `-54.595306`
  - max `65.140816`
  - peak `65.140816`
  - RMS `15.76966`
  - clip_fraction `0.9492142`

Conclusion:

- The 65x clipped waveform is not caused only by Rust talker/code-predictor
  token divergence.
- The audio codec decoder is independently wrong: correct Python codec tokens
  still decode to a clipped Rust waveform.
- Next priority is audio codec decoder parity against the Rust references
  (`/tmp/qwen3-tts-rs/qwen_tts/src/audio/tokenizer/v2`) and Python eager
  decoder output. Keep talker/code-predictor debug paused until decoder output
  scale is fixed.

## 2026-05-27 Audio Codec SnakeBeta Fix

Reference comparison:

- Python
  `/opt/miniconda3/lib/python3.13/site-packages/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
  `SnakeBeta.forward` uses:
  - `alpha = exp(alpha)`
  - `beta = exp(beta)`
  - `x + sin(x * alpha)^2 / (beta + 1e-9)`
- Rust reference
  `/tmp/qwen3-tts-rs/qwen_tts/src/audio/tokenizer/v2/snake_beta.rs`
  does the same and casts to F32 for the sin computation.

Bug:

- Our `tts_rs_qwen_burn/src/shared/nn/activation.rs` implementation used raw
  `alpha` and raw `beta` directly and used `1e-8`.
- This makes the vocoder activation numerically wrong and explains the huge
  unclamped waveform scale.

Fix:

- Changed `AudioCodecSnakeBeta::forward` to:
  - cast input and parameters to F32,
  - apply `exp()` to `alpha` and `beta`,
  - use epsilon `1e-9`,
  - cast back to the input dtype.

Verification:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e \
  rust_audio_codec_decodes_python_eager_codes_without_clipping -- --ignored --nocapture
```

- Passed.
- This proves the Rust audio codec no longer clips when fed Python eager codec
  groups.

Current CLI E2E check:

```bash
cargo run --release -p tts_rs_qwen_burn --bin qwen3-tts -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64
```

Result:

- Wrote `./0000.wav` and `./manifest.json`.
- Manifest waveform stats:
  - min `-0.15015113`
  - max `0.1389156`
  - peak `0.15015113`
  - RMS `0.051504903`
  - clip_fraction `0.0`
  - verdict `plausible`

Conclusion:

- End-to-end Rust inference now produces a non-clipped plausible audio file in
  the current folder.
- Remaining talker/code-predictor exact alignment issues still exist, but they
  are no longer masking the audio codec scale failure.

## 2026-05-27 Audio Codec Output Clamp Parity

Reference comparison:

- Python `Qwen3TTSTokenizerV2Decoder.forward` returns
  `wav.clamp(min=-1, max=1)`.
- Rust reference `/tmp/qwen3-tts-rs/qwen_tts/src/audio/decoder/v2.rs` also
  clamps after the final conv.
- Our Rust decoder returned raw `h` from the final decoder entry and only
  clamped in the WAV writer.

Change:

- Added `h.clamp_min(-1.0).clamp_max(1.0)` at the end of
  `Qwen3TtsAudioCodecDecoder::forward`.

Verification:

```bash
cargo test --release -p tts_rs_qwen_burn --test alignment_e2e \
  rust_audio_codec_decodes_python_eager_codes_without_clipping -- --ignored --nocapture
```

- Passed after the clamp parity change.

CLI recheck:

```bash
cargo run --release -p tts_rs_qwen_burn --bin qwen3-tts -- \
  --model-dir Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用语音合成。" \
  --language Chinese \
  --speaker Vivian \
  --output-dir . \
  --max-new-tokens 64
```

- Still writes `./0000.wav`.
- Manifest remains non-clipped:
  - peak `0.15015113`
  - RMS `0.051504903`
  - clip_fraction `0.0`
  - verdict `plausible`
