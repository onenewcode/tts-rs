# Qwen3-TTS Base / CustomVoice 开发方案

## 1. 背景与目标

本期只实现并对齐两条能力线：

- `CustomVoice`
- `Base`

目标是让 `tts-rs` 在现有三 crate 结构下，补齐上游 `qwen_tts` 与 `qwen3-tts-rs` 已公开的核心调用能力，并保持 `tts_cli` 仍然是薄封装。

本期不包含：

- `VoiceDesign`
- CLI 批量输入
- Base 模型上的正式 `instruct` 能力
- 远程 URL 参考音频下载

## 2. 当前仓库现状

### 2.1 已具备能力

- `tts_cli` 已支持：
  - `synthesize base`
  - `synthesize custom-voice`
- `CustomVoice` 当前已支持：
  - `text`
  - `language`
  - `speaker`
- `tts_qwen3_tts` 已有：
  - package/model-dir 加载
  - talker + audio codec decoder 推理
  - request compiler
  - ignored real-model smoke 框架

### 2.2 已确认限制

- generator 当前仅支持 `batch size = 1`
- `CustomVoice` 尚未支持 `instruct`
- `Base` 尚未支持：
  - `ref_audio`
  - `ref_text`
  - `x_vector_only`
  - 可复用 clone prompt
- audio codec encoder 权重已可加载，但尚未接入“参考音频 -> clone conditioning”链路

### 2.3 验证状态

- `CustomVoice`
  - 已有本地可运行 smoke 基线
  - 当前仅验证 `text/language/speaker` 路径
- `Base`
  - 当前在本仓库里还没有完成 voice-clone 路径的端到端验证
  - 因此本期 Base 方案必须区分：
    - `按上游接口对齐的设计`
    - `仓库内已 smoke 确认的行为`

## 3. 能力范围与行为定义

### 3.1 CustomVoice

对齐目标：

- 单条调用支持：
  - `text`
  - `language`
  - `speaker`
  - `instruct` 可选
- Rust API 提供顺序 batch 包装能力，但底层仍逐条串行执行

行为约束：

- `speaker` 不能为空
- `instruct` 可为空；为空时与现有行为一致
- `speaker`、`instruct` 仅允许用于 `CustomVoice` 请求
- `CustomVoice` 请求禁止携带 Base clone 相关字段

### 3.2 Base

对齐目标：

- 单条调用支持：
  - `text`
  - `language`
  - `ref_audio` 本地 WAV
  - `ref_text` 可选
  - `x_vector_only` 布尔开关
  - `create_voice_clone_prompt(...)` 复用 prompt
- Rust API 提供顺序 batch 包装能力，但底层仍逐条串行执行

行为约束：

- `Base` clone 输入只支持本地 WAV 文件
- `ref_audio` 与预构建 clone prompt 互斥
- `x_vector_only = false` 且提供 clone 参考时，`ref_text` 必须可提供
- `x_vector_only = true` 时允许没有 `ref_text`
- `Base` 请求禁止携带 `speaker` 或 `instruct`

### 3.3 批量能力

本期仅在 Rust API 层支持 batch：

- 新增 `Qwen3TtsEngine::synthesize_batch(...)`
- 输入为多条请求，内部按顺序逐条 synthesize
- 输出顺序必须与输入顺序一致
- 任何单条失败时，默认立即返回该错误，不继续后续请求

CLI 本期保持单条请求模式，不增加 JSON/YAML batch 文件入口。

## 4. 目标 API 设计

### 4.1 `tts_qwen3_tts` 公共请求类型

#### `CustomVoiceRequest`

在现有字段基础上新增：

- `instruct: Option<String>`

#### `BaseRequest`

在现有字段基础上新增可选 clone 条件：

- `voice_clone: Option<BaseVoiceCloneConditioning>`

其中：

- `BaseVoiceCloneConditioning::ReferenceAudio`
  - `path: PathBuf`
  - `transcript: Option<String>`
  - `x_vector_only: bool`
- `BaseVoiceCloneConditioning::Prompt`
  - `Qwen3TtsVoiceClonePrompt`

#### `Qwen3TtsVoiceClonePrompt`

该结构代表已准备好的、可重复用于多个 Base 请求的 clone conditioning。

最小需要包含：

- clone prompt 的 codec/semantic prompt token 序列
- 可选 transcript
- prompt 的模式标记（ICL / x-vector-only）

### 4.2 `Qwen3TtsEngine` 公共方法

新增：

- `create_voice_clone_prompt(...) -> Result<Qwen3TtsVoiceClonePrompt, Qwen3TtsError>`
- `synthesize_batch(...) -> Result<Vec<PcmAudio>, Qwen3TtsError>`

保留：

- `synthesize(...)`

### 4.3 `tts_cli`

新增参数：

- `synthesize custom-voice`
  - `--instruct <TEXT>`
- `synthesize base`
  - `--ref-audio <PATH>`
  - `--ref-text <TEXT>`
  - `--x-vector-only`

CLI 仍然一条请求对应一个输出 WAV。

## 5. 编译与 Prompt Recipe 设计

### 5.1 Prompt Recipe 列表

编译器需要显式区分以下 recipe：

- `BasePlain`
- `BaseVoiceCloneIcl`
- `BaseVoiceCloneXVectorOnly`
- `CustomVoicePlain`
- `CustomVoiceInstructed`

### 5.2 Prompt 语义原则

实现时遵循两条原则：

- 尽量靠近上游 Qwen 已公开的接口意图
- 不在未验证前把 prompt 细节宣传为“已确认最佳”

因此文档里把 prompt 分为：

- `接口语义必须一致`
- `实际文本模板允许在实现期微调`

### 5.3 Base clone 条件拼接

Base clone 需要把参考音频编码结果转为可注入 talker 的 prompt token。

本期采用的最小方案：

- 参考音频经过 audio codec encoder
- 提取可作为 talker codec prefix 的 semantic prompt token
- ICL 模式额外带入 `ref_text`
- x-vector-only 模式不要求 `ref_text`

## 6. 参考音频处理链路

### 6.1 输入范围

仅支持本地 WAV。

支持的 WAV 读取要求：

- PCM int16 / int24 / int32
- float32
- mono 或 multi-channel

### 6.2 预处理流程

1. 读取 WAV
2. 校验存在有效样本
3. 多声道混合为单声道
4. 重采样到 codec encoder 期望采样率
5. 送入 audio codec encoder
6. 生成 clone conditioning / voice clone prompt

### 6.3 失败策略

以下情况必须返回明确错误：

- 文件不存在
- WAV 头非法或格式不支持
- 空音频
- 重采样后样本为空
- 编码后没有可用 prompt token

## 7. 模型能力识别

为了避免把错误请求发给错误模型，需要在加载时识别模型类型。

### 7.1 识别优先级

优先依据：

- model dir / package name 中是否包含：
  - `customvoice`
  - `base`

兜底依据：

- `spk_id` 是否为空

### 7.2 识别结果用途

- `CustomVoice` 模型：允许 `speaker/instruct`
- `Base` 模型：允许 `voice clone`
- 不匹配时返回显式 `InvalidInput`

## 8. 实现分期

### Phase 1: 文档落地

- 新增本方案文档
- 更新测试文档中的验证状态与 smoke 标准

### Phase 2: Public API 与 CLI 入口

- 扩展 request types
- 扩展 engine helper methods
- 扩展 CLI 参数解析

### Phase 3: Compiler / Prompt Recipe

- 接入 `CustomVoice.instruct`
- 接入 Base clone recipe
- 增加模型能力校验

### Phase 4: Reference Audio Runtime

- 本地 WAV 读取
- mixdown / resample
- audio codec encoder 前向
- clone prompt 生成

### Phase 5: 测试与 smoke

- 单元测试
- CLI 解析测试
- ignored real-model smoke
- Base 手工 smoke 命令验证

## 9. 验收标准

本节为本期必须满足的正式验收标准。未满足任一 `阻塞项`，该阶段不可视为完成。

### 9.1 文档验收标准

必须满足：

- 已存在 `docs/qwen3_tts_base_customvoice_plan.md`
- 文档明确区分：
  - 实现范围
  - 非范围内容
  - 当前已验证状态
  - Base 未验证风险
  - 验收标准
- `docs/testing_tts_qwen.md` 已补充 Base / CustomVoice 验证状态

阻塞项：

- 缺少 Base 未验证说明
- 缺少明确 smoke 命令模板
- 验收标准仅写“能跑通”而没有具体条件

### 9.2 API 验收标准

必须满足：

- `CustomVoiceRequest` 可表达 `instruct`
- `BaseRequest` 可表达 `ref_audio/ref_text/x_vector_only` 或已构建 prompt
- `Qwen3TtsEngine::create_voice_clone_prompt(...)` 可用
- `Qwen3TtsEngine::synthesize_batch(...)` 可用
- 新增接口不破坏现有最小单条请求调用方式

阻塞项：

- 需要用户在调用时手动拼内部 token
- 旧的 `BaseRequest::new(...)` / `CustomVoiceRequest::new(...)` 行为被破坏
- API 无法区分 raw reference audio 与 prepared prompt

### 9.3 CLI 验收标准

必须满足：

- `custom-voice --instruct` 解析通过
- `base --ref-audio` 解析通过
- `base --ref-text` 解析通过
- `base --x-vector-only` 解析通过
- 非法组合报错可读

阻塞项：

- 参数冲突时静默忽略
- 参数解析成功但请求映射错误
- 需要额外隐藏环境变量才能使用新能力

### 9.4 编译器 / Prompt 验收标准

必须满足：

- 能区分 5 类 recipe
- `CustomVoice.instruct` 走独立 recipe
- Base clone 能把 reference conditioning 注入编译结果
- 错误模型 + 错误请求组合会被拒绝

阻塞项：

- `instruct` 只是 CLI 接收但编译器完全丢弃
- Base clone 请求没有进入独立 recipe
- Base / CustomVoice 模型能力边界未校验

### 9.5 参考音频链路验收标准

必须满足：

- 本地 WAV 可以被读取
- 多声道能混成单声道
- 非 24k 音频能被重采样到目标采样率
- encoder 能产出非空 clone prompt token
- clone prompt 可复用到多次 synthesize

阻塞项：

- 只支持已经是 24k mono 的理想输入
- 参考音频一旦重采样就崩溃或失真到不可用
- create prompt 成功但无法再用于 synthesize

### 9.6 测试验收标准

必须满足以下自动化测试层：

- `cargo test -p tts_qwen3_tts --lib`
- `cargo test -p tts_qwen3_tts --tests --no-run`
- `cargo test -p tts_cli --lib`

至少需要新增覆盖：

- request 默认值与非法组合
- prompt recipe 选择
- CLI 参数解析
- Base clone 输入校验
- clone prompt 复用
- batch wrapper 顺序与错误传播

阻塞项：

- 只有手工验证，没有自动化测试
- 只有 happy path，没有非法组合测试
- Base clone 没有任何测试覆盖

### 9.7 CustomVoice smoke 验收标准

CustomVoice 视为“已验证完成”，至少要满足：

- 使用本地可用 `CustomVoice` 权重
- `speaker` 指定后可生成非空 WAV
- `instruct` 指定后同样可生成非空 WAV
- 输出满足：
  - mono
  - 24000 Hz
  - 16-bit PCM
  - 非零帧数

建议 smoke 命令模板：

```bash
cargo run --release -p tts_cli -- synthesize custom-voice \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0___6B-CustomVoice \
  --text '你好，欢迎使用 tts-rs 自定义音色测试。' \
  --language Chinese \
  --speaker Vivian \
  --instruct '用特别愤怒的语气说' \
  --backend flex \
  --max-new-tokens 100 \
  --output ./tts_cli_custom_voice_instruct_smoke.wav
```

阻塞项：

- `instruct` 加上后反而无法生成
- 输出 WAV 文件存在但为空音频
- 生成只在某个测试分支里可用，CLI 主路径不可用

### 9.8 Base smoke 验收标准

Base 只有在以下条件全部满足后，才可以从 `unverified` 升级为 `repo-verified`：

- 使用本地可用 `Base` 权重
- 使用本地参考音频 WAV
- `ref_audio + ref_text` 路径可生成非空 WAV
- `x_vector_only` 路径至少能成功跑通一次
- 输出满足：
  - mono
  - 24000 Hz
  - 16-bit PCM
  - 非零帧数
- ignored smoke 已添加且可本地执行
- 手工 CLI smoke 命令已确认跑通

建议 smoke 命令模板：

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text 'I am solving the equation: x = [-b ± √(b²-4ac)] / 2a.' \
  --language English \
  --ref-audio ./fixtures/base_clone_ref.wav \
  --ref-text 'Okay. Yeah. I resent you. I love you.' \
  --backend flex \
  --max-new-tokens 100 \
  --output ./tts_cli_base_clone_smoke.wav
```

x-vector-only 命令模板：

```bash
cargo run --release -p tts_cli -- synthesize base \
  --model-dir ./Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text 'Hello from the Base voice clone x-vector-only smoke path.' \
  --language English \
  --ref-audio ./fixtures/base_clone_ref.wav \
  --x-vector-only \
  --backend flex \
  --max-new-tokens 100 \
  --output ./tts_cli_base_xvector_smoke.wav
```

阻塞项：

- 只完成代码实现，但没有 Base 本地 smoke
- ICL 路径能跑，x-vector-only 路径完全不可用
- CLI 能解析，但运行时无法产出有效 WAV

### 9.9 发布前结论标准

发布或对外宣称完成前，结论必须按以下口径输出：

- 如果只有代码与单测完成，但 Base smoke 未完成：
  - 只能写 `Base support implemented but not yet repo-verified`
- 如果 CustomVoice smoke 完成、Base 未完成：
  - 必须分别汇报，不得统称“Base + CustomVoice 已完成”
- 只有当 Base 与 CustomVoice 都通过各自 smoke 后：
  - 才能写 `Base + CustomVoice repo-verified`

## 10. 风险与默认决策

### 10.1 已知风险

- Base clone 路径当前没有仓库内已知成功基线
- audio codec encoder 前向实现可能比 decoder 侧更容易出现对齐偏差
- 上游 prompt 细节若未完全复刻，可能导致效果弱于官方实现

### 10.2 默认决策

- batch 仅做 Rust API 串行包装
- 参考音频仅支持本地 WAV
- Base 的 `x_vector_only` 先以实验能力实现并验证
- Base 在 smoke 跑通前统一标记为 `unverified` 或 `experimental`


## 11. 交付物清单

本期交付物必须至少包含以下内容：

- 开发文档：`docs/qwen3_tts_base_customvoice_plan.md`
- 测试与验收文档更新：`docs/testing_tts_qwen.md`
- API 层改动：
  - `CustomVoiceRequest` 新能力
  - `BaseRequest` clone 条件表达
  - `Qwen3TtsEngine` 新 helper
- CLI 层改动：
  - `--instruct`
  - `--ref-audio`
  - `--ref-text`
  - `--x-vector-only`
- 自动化测试：
  - request / compiler / CLI / clone prompt / batch wrapper
- ignored smoke：
  - `CustomVoice` instruct smoke
  - `Base` clone smoke

缺少任一项时，默认视为“实现未完成”。

## 12. 验收记录模板

每次阶段性完成后，验收记录至少要包含：

- 日期
- 变更范围
- 使用模型
- 使用 backend
- 执行命令
- 结果状态：`pass` / `fail` / `blocked`
- 产物路径
- 备注：
  - 是否为 `repo-verified`
  - 是否仍是 `unverified`
  - 是否存在已知偏差

建议记录格式：

```text
Date:
Scope:
Model:
Backend:
Commands:
Result:
Artifacts:
Verification level:
Notes:
```

## 13. Base 验收升级规则

`Base` 的验收状态分为三个等级：

- `designed`
  - 文档、接口、测试计划已完成
  - 尚未证明仓库内功能可用
- `implemented-but-unverified`
  - 代码、单测、CLI 参数已完成
  - 但本地 Base smoke 未跑通
- `repo-verified`
  - 代码完成
  - 自动化测试通过
  - 本地 Base smoke 跑通
  - 手工 CLI 验证跑通

升级条件：

- `designed -> implemented-but-unverified`
  - 代码实现完成
  - 至少完成 request/compiler/CLI 自动化测试
- `implemented-but-unverified -> repo-verified`
  - `ref_audio + ref_text` 路径成功
  - `x_vector_only` 路径成功
  - WAV 产物满足格式要求
  - 验收记录已补齐

降级条件：

- 如果后续回归导致 Base smoke 失败，状态必须从 `repo-verified` 回退到 `implemented-but-unverified`，直到重新验证通过。

## 14. CustomVoice 验收升级规则

`CustomVoice` 的验收状态分为两个等级：

- `implemented`
  - 代码与自动化测试完成
- `repo-verified`
  - `speaker` 路径 smoke 成功
  - `speaker + instruct` 路径 smoke 成功
  - WAV 产物满足格式要求

如果 `instruct` 路径失败，则本期 `CustomVoice` 只能算部分完成，不能宣称整体验收通过。
