# tts_cli 使用文档

`tts_cli` 是这个工作区的命令行入口。它负责参数解析、日志与输出路径处理，
真正的请求整理和模型执行则交给 `tts_app`。

这份文档只描述当前 `tts_cli` crate 的真实行为，示例与现有 `--help` 输出
保持一致，不包含已经过期的命令行参数。

## 功能概览

`tts_cli` 目前提供一个顶层工作流：

```bash
tts_cli synthesize <profile> [options]
```

当前支持的合成 profile：

- `base`：基础 Qwen3-TTS 合成，以及基于 base 模型的声音克隆
- `custom-voice`：使用 custom-voice 检查点，按指定 speaker 合成，并可选
  使用风格指令

CLI 输出为 PCM WAV 文件；如果输出目录不存在，会自动创建父目录。

## 前置条件

- 已安装 Rust 与 Cargo
- 本地已有 Qwen3-TTS 模型目录
- 建议使用 release 模式；debug 模式会明显更慢

当前仓库里常见的本地模型目录：

- `./Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

## 构建与运行

推荐用 release 模式运行：

```bash
cargo run --release -p tts_cli -- synthesize --help
```

如果你需要切换非默认后端，请通过 Cargo feature 选择，而不是使用命令行参数。
例如：

```bash
cargo run --release -p tts_cli --no-default-features --features cuda -- synthesize --help
```

注意：当前 CLI 已经不支持 `--backend`。有些旧文档可能还保留这个参数，但
现在 `tts_cli` 的运行后端是通过 crate feature 决定的，例如 `flex`、
`fusion`、`cuda`、`wgpu`、`metal`、`vulkan` 等。

## 命令结构

```bash
tts_cli synthesize base [OPTIONS] --text <TEXT> --output <OUTPUT>
tts_cli synthesize custom-voice [OPTIONS] --text <TEXT> --output <OUTPUT> --speaker <SPEAKER>
```

两个 profile 共有的参数：

- `--model-dir <MODEL_DIR>`：模型目录路径
- `--manifest <MANIFEST>`：用于非标准目录布局的 `qwen3_tts_package.yaml`
- `--text <TEXT>`：要合成的目标文本
- `--language <LANGUAGE>`：语言名，默认 `auto`
- `--output <OUTPUT>`：输出 WAV 路径
- `--sampling <SAMPLING>`：当前只支持 `greedy`
- `--log-level <LOG_LEVEL>`：`error`、`warn`、`info`、`debug`、`trace`
- profiling 参数：`--profiling`、`--profiling-per-step`、
  `--profiling-stage-summary`、`--no-profiling-stage-summary`、
  `--profiling-log-topk`

使用规则：

- `--model-dir` 和 `--manifest` 二选一
- `--model-dir` 应直接指向模型文件夹本身，不要指向它的父目录
- 语言名会按模型元数据做不区分大小写匹配

## base 模式

`base` 既可以做普通合成，也可以做基于参考音频的声音克隆。

### 基础合成

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the base profile." \
  --language English \
  --output ./out/base_plain.wav
```

### 使用 `ref_audio + ref_text` 的 ICL 声音克隆

这是 in-context learning 的声音克隆路径。`--ref-text` 必须是 `--ref-audio`
对应的真实文本转写，而不是你要生成的目标文本。

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone ICL smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --ref-text "Hello from the generated reference clip." \
  --output ./out/base_clone_icl_release.wav
```

### 使用 `--x-vector-only` 的声音克隆

这个模式只使用参考音频中的 speaker embedding。

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello from the Base voice clone x-vector-only smoke path." \
  --language English \
  --ref-audio ./out/base_reference_custom_voice.wav \
  --x-vector-only \
  --output ./out/base_clone_xvector_release.wav
```

参考音频相关规则：

- `--ref-text` 必须和 `--ref-audio` 一起使用
- `--x-vector-only` 必须和 `--ref-audio` 一起使用
- `--x-vector-only` 不能和 `--ref-text` 同时使用
- 如果想让 ICL clone 更稳定，最好提供参考音频的真实逐字稿

## custom-voice 模式

`custom-voice` 用于带 speaker 列表的 custom-voice 检查点。

### 基础 custom-voice 合成

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，欢迎使用 tts-rs。" \
  --language Chinese \
  --speaker Vivian \
  --output ./out/custom-voice.wav
```

### 带 `--instruct` 的 custom-voice 合成

`--instruct` 用来描述目标文本应该采用的说话风格。

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --output ./out/custom-voice-instruct.wav
```

当前仓库里的 custom-voice 模型常见 speaker 包括：

- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ryan`
- `Aiden`
- `Ono_Anna`
- `Sohee`

实际可用的 speaker 以模型元数据为准；如果传入不支持的 speaker，CLI 会报出
当前模型支持的值。

## manifest 模式

大多数场景优先使用 `--model-dir`。只有当模型文件布局不是默认目录结构时，
才需要使用 `--manifest`。

示例：

```bash
cargo run --release -p tts_cli -- synthesize base \
  --manifest ./path/to/qwen3_tts_package.yaml \
  --text "Hello from a custom manifest layout." \
  --language English \
  --output ./out/base_manifest.wav
```

## 常见问题

`unexpected argument '--backend' found`

- 当前 CLI 已不再接受 `--backend`
- 请改为通过 Cargo feature 选择后端

`the following required arguments were not provided`

- 检查是否漏掉了 `--text`、`--output` 或 `--speaker`
- 对 `custom-voice` 来说，`--speaker` 是必填项

`--ref-text is required when --ref-audio is used without --x-vector-only`

- base 模式下的 ICL clone 需要同时提供参考音频和对应转写文本

`unsupported speaker`

- `--speaker` 的值必须存在于加载后的模型元数据里
- base 模型通常不会提供命名 speaker 列表

启动或生成速度慢

- 请使用 `--release`
- 大模型加载阶段可能需要明显时间，尤其是首次运行时

## 常用帮助命令

```bash
cargo run --release -p tts_cli -- synthesize --help
cargo run --release -p tts_cli -- synthesize base --help
cargo run --release -p tts_cli -- synthesize custom-voice --help
```
