# Qwen3-TTS Load Refactor (B 方案讨论稿)

## 1. 文档定位

这是一份**讨论稿**，不是最终定稿。

这份文档只落实当前已经明确的方向，不再扩展到量化，不再为量化预留口。

当前已确认的边界：

- 第一优先目标：
  - 稳态内存
  - 加载架构整洁度
- 重构激进程度：
  - 允许删改现有内部边界和文件布局
- 实施方式：
  - 接受 staged migration，先引入新链路，再删旧链路
- dtype 范围：
  - 只保留 `f32` / `f16` / `bf16`

本稿落实的不是“把 cast 提前一点”的温和方案，而是 **B 方案：按运行意图拆 resident runtime 形态**。

---

## 2. 为什么选 B 方案

之前不够激进的方案有一个共同问题：

- 它们默认仍然围绕一个“全量 loaded model”转
- 只是把 dtype 转换从 post-load 挪到了 load-time
- 但**没有动到真正导致稳态内存和结构混乱的根**

当前实现的真正问题，不是单纯“多做了一次 cast”，而是：

1. **加载是 request-agnostic 的**
   - 先加载一个“大而全”的 resident runtime
   - 然后再决定这次请求到底是 base、custom-voice、voice-clone 还是 prompt 生成
2. **resident runtime 形态只有一种**
   - talker / codec / speaker 被捆成一个大块
   - 即使这次请求根本不需要 speaker encoder，也可能被一起带进内存
3. **execution 在承担 load 的编排职责**
   - `execution/loaded_model.rs` 现在既像 runtime wrapper，又像 load assembler
4. **model 层还在处理权重文件导入**
   - `model/*/weights.rs` 让 `model/` 同时承担“模型定义”和“文件加载”

如果主要目标是稳态内存和架构整洁度，那么**最值得打掉的不是某个 cast 函数，而是“一种 runtime 打天下”的前提**。

所以 B 方案的核心不是“重新组织 loader 文件”，而是：

> **让 runtime 的常驻形态跟运行意图绑定，而不是跟 checkpoint 的全集能力绑定。**

---

## 3. 当前实现的具体问题

### 3.1 全量 runtime 形态过粗

当前内部常驻对象本质上是：

- talker
- decoder / codec
- speaker encoder（如果配置里有）
- compiler

它们被装进一个统一的 `Qwen3TtsModelInner`，再包进 `Qwen3TtsLoadedModel`。

这个结构的问题是：

- custom-voice 合成不需要 speaker encoder
- 纯 prompt 创建不需要 talker
- 但 runtime 形态并没有反映这些差异

换句话说，当前系统的 resident runtime 是按“模型理论能力全集”组织的，而不是按“本次操作真正需要什么”组织的。

### 3.2 `tts_app` 的生命周期特点没有被利用

当前 `tts_app::QwenAppService::synthesize_prepared(...)` 的路径是：

- 按本次请求准备 request
- load model
- synthesize
- close handle
- remove handle

这说明当前主流程本来就是**按请求加载、按请求销毁**。

既然不是长期驻留的通用服务进程，就没有必要坚持“所有请求都加载同一种 superset runtime”。

这恰恰说明 B 方案是合理的：  
**加载就应该感知意图。**

### 3.3 capability 和 resident runtime 耦合过重

当前 capability 投影还依赖 `Qwen3TtsLoadedModel`。

这会带来两个问题：

- capability 判定和 resident runtime 形态绑死
- 后续如果拆 runtime 形态，就会被现有 capability 依赖链反向拖住

事实上，绝大多数 capability 信息都应该来自：

- package 元数据
- compiler 解析结果
- speaker encoder 配置是否存在

而不是来自“一个已经完整加载好的执行对象”。

### 3.4 dtype 仍是附着在大 runtime 上的后处理概念

即使把 cast 前移，如果系统仍然坚持“先决定一个大 runtime，再全量导入”，那么它仍然没有解决：

- 为什么要加载这块模块
- 哪些模块这次根本不应该进 resident 内存

所以 dtype 前移只是必要条件，不是主方案本身。

---

## 4. B 方案的核心设计

### 4.1 设计原则

B 方案只坚持一个总原则：

> **按运行意图构建最小 resident runtime。**

也就是：

- 不是“先有一个完整 runtime，再去跑不同请求”
- 而是“先知道这次要做什么，再加载最小足够运行的 runtime”

### 4.2 引入一等公民：Load Intent

新的 load 系统不再只接收 `package + dtype`，而必须接收**加载意图**。

内部新增：

```rust
pub(crate) enum LoadIntent {
    BaseSynthesis,
    BaseVoiceCloneSynthesis,
    CustomVoiceSynthesis,
    VoiceClonePrompt,
}
```

这个 `LoadIntent` 不是可选优化，而是整个 B 方案成立的前提。

#### 每种意图对应的最小模块集合

| LoadIntent | 必需模块 |
|---|---|
| `BaseSynthesis` | talker + codec + compiler |
| `BaseVoiceCloneSynthesis` | talker + codec + compiler + speaker encoder |
| `CustomVoiceSynthesis` | talker + codec + compiler |
| `VoiceClonePrompt` | codec + speaker encoder + compiler |

关键点：

- `VoiceClonePrompt` 不需要 talker
- `CustomVoiceSynthesis` 不需要 speaker encoder
- `BaseVoiceCloneSynthesis` 才真正需要 speaker encoder

这才是 B 方案对稳态内存最直接的改动。

---

## 5. 新的 runtime 形态

### 5.1 旧形态：一个大而全的 loaded model

当前相当于：

```text
LoadedModel
  = talker + codec + speaker? + compiler + device
```

### 5.2 新形态：按意图拆 runtime

新的 resident runtime 不再只有一种 struct，而是改成**分形态 runtime**。

建议引入如下对象：

```rust
pub(crate) struct CoreSynthesisRuntime<B: Backend> {
    pub device: B::Device,
    pub compiler: Qwen3TtsRequestCompiler,
    pub talker: LoadedTalker<B>,
    pub codec: LoadedCodec<B>,
}

pub(crate) struct VoiceCloneSupportRuntime<B: Backend> {
    pub device: B::Device,
    pub compiler: Qwen3TtsRequestCompiler,
    pub codec: LoadedCodec<B>,
    pub speaker: LoadedSpeaker<B>,
}

pub(crate) enum LoadedRuntime<B: Backend> {
    BaseSynthesis(CoreSynthesisRuntime<B>),
    BaseVoiceClone {
        core: CoreSynthesisRuntime<B>,
        speaker: LoadedSpeaker<B>,
    },
    CustomVoice(CoreSynthesisRuntime<B>),
    VoiceClonePrompt(VoiceCloneSupportRuntime<B>),
}
```

这里的重点不是名字，而是语义：

- runtime shape 必须显式表达“本次到底加载了什么”
- execution 不再默认所有路径都能访问同一组字段

这会强迫实现层真正遵守最小加载原则。

### 5.3 不再允许 `Option<speaker_encoder>` 这种“半裁剪”结构

旧结构里：

- `speaker_encoder: Option<_>`

这个写法表面上是“可选”，实际上仍然把“完整 runtime”当默认形态。

B 方案要求去掉这种做法。

原因：

- `Option` 只是在大 runtime 上打洞
- 它不会改变 load orchestration 的中心思想
- 它也不会让 execution 代码自然分裂成按意图运行的路径

所以 B 方案要求：

- 不是一个 runtime 里 `Option<speaker>`
- 而是从类型层面区分不同 runtime 形态

---

## 6. 新的加载架构

### 6.1 加载分成两阶段

新的 loading 子系统分两阶段：

#### 阶段 A：轻量解析

只做元数据和编译准备，不加载大权重：

- package 归一化
- generation config 读取
- compiler 构建
- speaker encoder 配置存在性判断
- capability 所需的静态信息解析

输出：

```rust
pub(crate) struct ResolvedPackage {
    pub package: Qwen3TtsPackage,
    pub compiler: Qwen3TtsRequestCompiler,
    pub package_features: PackageFeatureFlags,
}
```

#### 阶段 B：按意图装配 resident runtime

输入：

- `ResolvedPackage`
- `LoadIntent`
- `LoadDTypePlan`

输出：

- `LoadedRuntime`
- `LoadReport`

### 6.2 入口应该从 request-aware 路径拿到 intent

当前系统准备 request 的地方在 `tts_app`。

这意味着 `tts_app` 已经知道：

- 当前是 base 还是 custom-voice
- 当前是否是 voice clone
- 当前是否只是创建 voice clone prompt

因此 B 方案要求：

- `tts_app` 在准备 request 后，同时生成 `LoadIntent`
- `tts_qwen3_tts` 的 load 不再是完全 request-agnostic

也就是说，后续内部接口要从：

```rust
Qwen3TtsEngineConfig { package, profiling, dtype }
```

演进为至少内部可表达：

```rust
InternalLoadConfig {
    package,
    profiling,
    dtype,
    intent,
}
```

这里不要求第一步就把所有公开 API 改掉，但内部装配链路必须先支持 intent。

---

## 7. 目录和职责重划

### 7.1 `model/` 禁止继续承载权重文件导入

`model/` 只保留：

- `config`
- `network`
- `infer`

它不再负责：

- safetensors store
- remapper
- dtype cast
- 文件路径处理

因此下面这些文件是 B 方案下的清理对象：

- `tts_qwen3_tts/src/model/talker/weights.rs`
- `tts_qwen3_tts/src/model/codec/weights.rs`
- `tts_qwen3_tts/src/model/speaker/weights.rs`

### 7.2 新的 `loading/` 结构

建议重组为：

```text
tts_qwen3_tts/src/loading/
  mod.rs
  package/
  intent.rs
  plan.rs
  report.rs
  runtime.rs
  store_adapter.rs
  subsystems/
    talker.rs
    codec.rs
    speaker.rs
```

各文件职责如下：

- `intent.rs`
  - `LoadIntent`
- `plan.rs`
  - `LoadDTypePlan`
- `report.rs`
  - `LoadReport`
- `store_adapter.rs`
  - float-only 的 load-time dtype 转换 adapter
- `subsystems/*`
  - talker / codec / speaker 的真正权重加载入口
- `runtime.rs`
  - 根据 `ResolvedPackage + LoadIntent + LoadDTypePlan` 组装 `LoadedRuntime`

### 7.3 `execution/` 只消费 runtime，不组装 runtime

`execution/` 的新职责是：

- request compile 后的运行
- session start / step / finish
- prompt 生成
- waveform finalize

`execution/` 不再知道：

- safetensors
- remapper
- package artifact 路径
- dtype adapter
- 哪些 subsystem 需要加载

---

## 8. dtype 策略

这轮范围收死，不留量化口。

### 8.1 公开 dtype 只保留三档

- `f32`
- `f16`
- `bf16`

删除：

- `f64`
- `flex32`
- 所有 `q*`

### 8.2 dtype 决策对象

加载层内部只保留：

```rust
pub(crate) struct LoadDTypePlan {
    pub runtime_float_dtype: FloatDType,
    pub tensor_dtype: DType,
}
```

规则：

- `F32 -> (F32, DType::F32)`
- `F16 -> (F16, DType::F16)`
- `BF16 -> (BF16, DType::BF16)`

### 8.3 dtype 转换位置

所有浮点权重都在 `load_from` 的 adapter 链中直接转成目标 dtype。

明确禁止：

- load 完成后整模型再做一次 cast
- execution 层对 resident weights 做 dtype rewrite

---

## 9. capability 的新来源

B 方案要求 capability 不再依赖“完整 loaded runtime 是否存在”。

### 9.1 capability 应来自元数据解析

`project_capabilities(...)` 应只依赖：

- package 元数据
- compiler profile 结果
- speaker encoder 配置存在性

而不是依赖：

- `Qwen3TtsLoadedModel`
- `supports_voice_clone()` 这种运行时包装器方法

### 9.2 为什么要这样改

因为一旦 runtime shape 按意图拆分：

- custom-voice runtime 本来就不会带 speaker
- 但这不代表 package 不支持 voice clone prompt

如果 capability 继续绑定某个具体 runtime shape，就会出现语义混乱：

- “这次没加载” 不等于 “这个包不支持”

因此 capability 必须回到 metadata analysis 阶段。

---

## 10. 实施顺序（staged migration）

因为你接受 staged migration，所以这里明确采用“三阶段切换”。

### 阶段 1：搭新骨架，不切主入口

新增但暂不完全接管：

- `LoadIntent`
- `LoadDTypePlan`
- `LoadedRuntime`
- `LoadReport`
- `loading/subsystems/*`
- `loading/runtime.rs`

这阶段允许旧入口暂时继续存在，但新路径已经能单独跑通。

### 阶段 2：让主入口转向 intent-aware runtime builder

把主装配链从旧路径切到：

```text
resolve package
-> derive intent
-> build runtime by intent
-> hand runtime to execution
```

这一步完成后：

- `execution/loaded_model.rs` 不再是 load 中心
- `model/*/weights.rs` 进入废弃状态
- capability 改由 metadata route 提供

### 阶段 3：删除旧形态

最终删除：

- `model/*/weights.rs`
- execution 中的 load orchestration
- 所有 post-load cast 逻辑
- 所有已废弃 dtype

---

## 11. 验收标准

### 11.1 架构验收

以下条件必须全部满足：

- `model/` 目录下不再存在 `weights.rs`
- `execution/` 目录不再承担 resident runtime 组装职责
- loading 子系统成为唯一合法的权重导入与 runtime 装配入口
- resident runtime 不再是单一 superset 形态
- 不再依赖 `Option<speaker_encoder>` 表达 runtime 裁剪

### 11.2 行为验收

按运行意图的最小加载集合应满足：

- `CustomVoiceSynthesis` 不加载 speaker encoder
- `BaseSynthesis` 不加载 speaker encoder
- `BaseVoiceCloneSynthesis` 加载 speaker encoder
- `VoiceClonePrompt` 不加载 talker

### 11.3 dtype 验收

- 只接受 `f32 / f16 / bf16`
- resident float weights 在 loader 返回时已经是目标 dtype
- 生产路径中不存在 post-load dtype rewrite

### 11.4 稳态内存验收

以相同机器、相同 backend、相同模型、相同请求测量：

- `rss_idle_after_load`
- `rss_idle_after_first_synthesis`

关注两类收益：

1. **dtype 收缩收益**
   - `f16` 和 `bf16` 应明显低于 `f32`
2. **runtime 形态裁剪收益**
   - `CustomVoiceSynthesis` 的常驻内存应低于 `BaseVoiceCloneSynthesis`
   - `VoiceClonePrompt` 不应承担 talker 常驻成本

也就是说，B 方案的稳态内存验收不只是比 `f32/f16/bf16`，还要比**不同意图形态之间是否真的裁剪成功**。

---

## 12. 风险与代价

### 12.1 主要收益

- 架构认知负担显著下降
- resident runtime 和业务意图对齐
- speaker encoder 的无效常驻可被消除
- 为后续更细的按需加载打基础

### 12.2 主要代价

- 内部接口会明显变化
- `tts_app` 与 `tts_qwen3_tts` 的耦合点会从“只传 request”变成“传 request + intent”
- 测试矩阵会从“一个 loaded model 跑所有路径”变成“多种 runtime shape 分别验证”

### 12.3 为什么仍然值得做

因为你的当前目标是：

- 稳态内存
- 架构整洁度

如果不把 resident runtime 形态本身拆开，只围绕一个 superset loaded model 修补，后面大概率还会回到同样的问题。

---

## 13. 本稿结论

B 方案的核心结论只有一句：

> **这轮重构不应把重点放在“优化一个全量 loaded model”，而应把重点放在“取消全量 loaded model 作为唯一常驻形态”。**

因此本轮推荐的主线不是：

- 继续围绕 `Qwen3TtsLoadedModel` 打补丁

而是：

- 引入 `LoadIntent`
- 拆分 resident runtime shape
- 让 loading 按意图装配最小运行时
- 让 execution 只消费已经成型的 runtime
- 彻底把 dtype 收口到 `f32/f16/bf16`

如果后续继续按这条线展开，下一版实施文档就应该开始把：

- 新对象列表
- 新文件边界
- staged migration 的每一步变更
- 具体测试矩阵

进一步写成可执行清单。
