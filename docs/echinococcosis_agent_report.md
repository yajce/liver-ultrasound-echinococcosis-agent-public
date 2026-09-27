# 基于多模态大模型与 Skill 的肝脏超声包虫病识别探索

本项目用约一周时间完成了一个小规模、可复现的探索性实验：整理肝脏超声图像，搭建统一的多模态 Agent 推理链路，并比较额外 Skill 对两种基础模型的影响。项目的重点不是证明系统已经具备临床诊断能力，而是观察同一类结构化判读提示是否会在不同模型上产生一致效果。

## 1. 研究背景与任务

项目最初包含四项任务：调研肝脏超声包虫病识别问题；构建可接入真实图像的多模态 Agent；尝试用 Skill 组织模型的观察与推理；在至少两个模型上比较 baseline 与 Skill 条件。最终纳入的模型为：

- GLM-5.3-Flash；
- DeepSeek-V4-Flash-Vision-Exp。

本项目实际回答的是一个较具体的问题：**Skill 对不同基础多模态模型的影响是否一致？** 这里的 Skill 是附加在统一任务说明之后的判读流程，不是新训练的分类器，也不是对模型参数进行微调。

实验采用二分类任务：`echinococcosis` 与 `negative`。对于明显不是肝脏超声或技术上完全无法解释的输入，模型还可以输出 `not_applicable`；解析失败则保留为 `parse_failed`。在已确认有效的 benchmark 中，后两种状态均按严格错误处理，而不会被强制转换为阳性或阴性。

## 2. 数据集与 benchmark 构建

### 2.1 阳性数据

阳性数据来自公开 supplementary PDF `Dataset2_US.pdf`。提取脚本使用 PyMuPDF 直接读取 PDF 内嵌图像对象，不把整页渲染成截图；内嵌 JPEG 尽量保持原始字节，其他图像格式才转换为 JPEG。根据预先确认的信息，流程排除了 Figure 25 对应的胸部 CT 和 Figure 85 对应的胸部 X-ray，并使用 SHA256 对其余图像去重。最终保留 83 张肝脏包虫病超声图像。

### 2.2 阴性数据

阴性数据来自 Annotated Ultrasound Liver images，原始图像分为 Normal、Benign、Malignant 三类。原始候选共 735 张，其中 Benign 200 张、Malignant 435 张、Normal 100 张。抽样前按文件名排序，再用固定随机种子 42 按原始比例抽取 83 张：

- Benign：23 张；
- Malignant：49 张；
- Normal：11 张。

这些阴性样本并不等同于“健康图像”。本项目中的 `negative` 表示“非包虫病肝脏超声”，其中包含正常肝脏，也包含良性和恶性肝脏病变。所有候选图和抽样结果均进行了 SHA256 重复检查，原图采用复制方式保留。

### 2.3 最终 benchmark 与固定划分

阳性 83 张和阴性 83 张合并后，使用固定随机种子 42 混合，得到 166 张图像。文件统一重命名为 `img_0001.jpg` 至 `img_0166.jpg`，文件名不包含标签信息；标签与来源追踪信息分别保存在独立 CSV 中。构建过程还检查了正、负来源之间的 SHA256 重复。

固定划分为：

- dev：100 张，50 positive / 50 negative；
- held-out test：66 张，33 positive / 33 negative。

dev 用于观察 Skill 的行为并迭代 v1 至 v4；held-out test 在 Skill 开发过程中不参与调参，只在所有实验方案冻结后用于最终评估。

Skill v4 有一项额外约束。其 worked reasoning example 由 dev 中一个样本的 source-provided morphology description 抽象而来，因此对应的 `img_0020.jpg` 被排除在 v4 的开发集评估之外。v4 的公平开发集 `dev_v4_eval` 共 99 张，包括 49 positive 和 50 negative；原始 dev100 没有被修改。

## 3. Agent 与实验框架

项目采用统一主链路：

```text
Image
  → Runner
  → Agent
  → ModelBackend
  → multimodal LLM
  → structured response
  → Parser
  → PredictionRecord / Evaluation
```

`ModelBackend` 定义统一的图像与提示词输入、文本响应、模型名称和调用耗时。GLM 与 DeepSeek 后端均通过 OpenAI-compatible 多模态消息格式发送本地图像，图像被编码为 Base64 data URL；两种模型共用相同的 Agent、任务 schema、Parser 和评估方式。

模型输出统一为一个 JSON 对象，包含：

- `image_valid`：图像是否可用于当前任务；
- `diagnosis`：`echinococcosis`、`negative` 或 `not_applicable`；
- `confidence`：模型自报置信度；
- `subtype`：可选分型；
- `findings`：模型对可见内容的简短描述。

Parser 保留 `parse_ok`，原始响应始终写入结果文件。Runner 每完成一张图就立即追加结果，并以 `image_id` 实现断点续跑；单次 API 失败不会伪造预测，也不会使整个批次丢失已经完成的记录。连接失败可以在同一运行目录中续跑，但模型预测错误不会成为重跑理由。

所有条件均使用 `baseline-v0.2` 的基础任务说明。baseline 并不表示模型“没有医学知识”，而是不给模型附加项目自定义的 echinococcosis Skill，只依赖模型预训练知识和统一任务说明。启用 Skill 时，Agent 从版本目录读取 `SKILL.md`，将文本附加到基础上下文，并在每条结果中单独记录 `skill_enabled`、`skill_version` 和 `skill_hash`；Skill 内容没有硬编码进 Python。

评估时，TP、FN、FP、TN 只使用有效二分类预测。严格准确率定义为 `(TP + TN) / split 全部样本数`，因此 `not_applicable` 和 `parse_failed` 在有效 benchmark 上都属于严格错误。`confidence` 仅表示模型 self-reported confidence，不是经过校准的概率。

## 4. Skill 设计与迭代

### 4.1 Skill v1：显式特征清单

v1 采用 observation-first 的结构化流程，同时列出较长的 CE morphology checklist，并加入 WHO-style 分型提示和 anti-overdiagnosis 约束。设计意图是避免模型只因“看到囊性病灶”就判断包虫病，同时让输出先描述观察、再整合证据。

实际 dev 结果显示，v1 不只是增加了说明文字，还改变了 GLM 的决策行为。与 baseline 相比，更多预测从 `echinococcosis` 转向 `negative` 或 `not_applicable`，specificity 提高，但 sensitivity 明显下降。文本分析中，`characteristic`、`double wall`、`hydatid sand`、`nonspecific` 等清单词汇在变化样本里更频繁出现。更谨慎的表述修正了一部分 false positive，也引入了更多 false negative。因此，与其简单说“v1 失败”，更准确的说法是：v1 将 GLM 推向了过度保守的决策方向。

### 4.2 Skill v2：缓解过度保守

v2 保留 v1 的总体结构，只对两个问题做定向调整：

- `not_applicable` 仅用于明确不是肝脏超声或技术上不可解释的图像；低对比度、非典型表现或病灶不明显本身不足以拒判；
- 未看见某一个典型特征，不能单独作为排除包虫病的证据；模型不应要求全部或多数清单特征同时出现。

这个修改降低了理论上的拒判和 checklist 门槛，但 GLM dev 的 sensitivity 没有恢复，严格准确率也未改善。说明核心问题不只是一两句过强措辞，而可能与整段显式清单如何改变证据组织方式有关。

### 4.3 Skill v3：minimal / ablation

v3 删除了长篇显式 CE checklist，只保留通用流程：

- image validity；
- systematic morphology observation；
- evidence integration；
- 同时避免过度诊断和过度排除；
- 沿用统一结构化输出。

这一版本测试的是“结构化观察本身”而不是“具体疾病特征清单”。在 GLM dev100 上，v3 的严格准确率回到 baseline 水平；迁移到 DeepSeek 后，v3 明显减少了 false positive，同时带来一定 sensitivity 损失。DeepSeek 的 paired dev 分析显示，15 个 wrong→correct 中有 13 个是 FP→TN，而 8 个 correct→wrong 中有 5 个是 TP→FN。这表明 v3 对 DeepSeek 主要起到较保守的决策修正作用。

### 4.4 Skill v4：加入 worked reasoning example

v4 在 v3 的最小结构上增加一个简短 worked reasoning example，示范：

```text
Observation → Evidence integration → Decision
```

示例强调先记录可见形态，再判断多个观察是否构成一致证据，最后做二分类决定。它由开发数据中的 source-provided morphology description 抽象、改写和泛化而来，没有直接把原始图像输入 Skill，也没有逐字复制原 caption，因此不应称为 multimodal few-shot。为减少单个阳性示例带来的默认阳性偏置，v4 紧接着加入 counterbalance，要求新图像只使用实际可见信息，不能强行模仿示例。

由于示例来源与 `img_0020.jpg` 有关，该样本没有计入 v4 的 99 张开发集评估。v4 是项目最后一个预先确定的 Skill 版本；最终 test 完成后不再继续调参或创建 v5。

## 5. Dev 阶段实验

开发阶段的结果用于决定下一版 Skill 的实验设计，不用于替代 held-out test 结论。下表中，v1、v2、v3 及相应 baseline 使用原始 dev100；v4 使用排除 `img_0020.jpg` 后的 dev99，两种口径必须分开理解。

| Model | Condition | Evaluation subset | N | TP | FN | FP | TN | NA | PF | Strict Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM | baseline | dev100 | 100 | 39 | 11 | 4 | 46 | 0 | 0 | 85.00% | 78.00% | 92.00% | 83.87% |
| GLM | Skill v1 | dev100 | 100 | 29 | 19 | 1 | 49 | 2 | 0 | 78.00% | 58.00% | 98.00% | 74.36% |
| GLM | Skill v2 | dev100 | 100 | 28 | 20 | 0 | 49 | 1 | 2 | 77.00% | 56.00% | 100.00% | 73.68% |
| GLM | Skill v3 | dev100 | 100 | 39 | 10 | 4 | 46 | 1 | 0 | 85.00% | 78.00% | 92.00% | 84.78% |
| GLM | Skill v4 | dev_v4_eval99 | 99 | 35 | 14 | 3 | 47 | 0 | 0 | 82.83% | 71.43% | 94.00% | 80.46% |
| DeepSeek | baseline | dev100 | 100 | 47 | 2 | 34 | 16 | 1 | 0 | 63.00% | 94.00% | 32.00% | 72.31% |
| DeepSeek | Skill v3 | dev100 | 100 | 44 | 6 | 24 | 26 | 0 | 0 | 70.00% | 88.00% | 52.00% | 74.58% |
| DeepSeek | Skill v4 | dev_v4_eval99 | 99 | 41 | 8 | 20 | 30 | 0 | 0 | 71.72% | 83.67% | 60.00% | 74.55% |

GLM 的演化路径很清楚：baseline 为 85.00%，v1 降至 78.00%，v2 进一步为 77.00%，v3 回到 85.00%。v1/v2 的 specificity 上升伴随着 sensitivity 从 78.00% 降至 58.00% 和 56.00%，显示显式 checklist 使模型更少判阳性，但代价过大。移除清单后的 v3 恢复了 baseline 水平。

v4 不能直接用 82.83% 与 dev100 的 85.00% 作完全同口径比较。过滤到同一 99 张后，公平对照为：

| Model | baseline | Skill v3 | Skill v4 |
|---|---:|---:|---:|
| GLM | 85.86% | 85.86% | 82.83% |
| DeepSeek | 62.63% | 69.70% | 71.72% |

在公平 dev99 上，GLM v4 低于 baseline 和 v3，并表现得更保守；DeepSeek v3、v4 则均高于其 baseline。DeepSeek 的主要变化仍来自 FP 减少：在相同 99 张上，FP 为 34→24→20，但 sensitivity 同时下降。以上是 Skill development evidence，只说明开发集上的行为方向，最终泛化判断必须以后续 held-out test 为准。

## 6. Held-out test 最终结果

最终结果使用冻结的 66 张 held-out test，其中阳性 33 张、阴性 33 张。六组结果覆盖完全相同的 `image_id`；GLM baseline 从原有 166 行结果中严格按 test split 过滤，其余五组各有 66 行。六组均无缺失、重复、额外 test 样本，且 `NA=0`、`PF=0`，因此本节的 Overall Accuracy 与 Strict Accuracy 相同。

为便于理解混淆矩阵：TP（阳判阳）表示真实阳性且预测阳性；FN（阳判阴）表示真实阳性但预测阴性；FP（阴判阳）表示真实阴性但预测阳性；TN（阴判阴）表示真实阴性且预测阴性。FNR 为 `FN / (TP + FN)`，FPR 为 `FP / (TN + FP)`。

| Model | Condition | N | TP | FN | FP | TN | Correct | Wrong | NA | PF | Accuracy | Precision | Sensitivity | Specificity | FNR | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | baseline | 66 | 29 | 4 | 0 | 33 | 62 | 4 | 0 | 0 | 93.94% | 100.00% | 87.88% | 100.00% | 12.12% | 0.00% | 93.55% |
| GLM-5.3-Flash | Skill v3 | 66 | 26 | 7 | 0 | 33 | 59 | 7 | 0 | 0 | 89.39% | 100.00% | 78.79% | 100.00% | 21.21% | 0.00% | 88.14% |
| GLM-5.3-Flash | Skill v4 | 66 | 22 | 11 | 0 | 33 | 55 | 11 | 0 | 0 | 83.33% | 100.00% | 66.67% | 100.00% | 33.33% | 0.00% | 80.00% |
| DeepSeek-V4-Flash-Vision-Exp | baseline | 66 | 32 | 1 | 18 | 15 | 47 | 19 | 0 | 0 | 71.21% | 64.00% | 96.97% | 45.45% | 3.03% | 54.55% | 77.11% |
| DeepSeek-V4-Flash-Vision-Exp | Skill v3 | 66 | 30 | 3 | 11 | 22 | 52 | 14 | 0 | 0 | 78.79% | 73.17% | 90.91% | 66.67% | 9.09% | 33.33% | 81.08% |
| DeepSeek-V4-Flash-Vision-Exp | Skill v4 | 66 | 30 | 3 | 9 | 24 | 54 | 12 | 0 | 0 | 81.82% | 76.92% | 90.91% | 72.73% | 9.09% | 27.27% | 83.33% |

### 6.1 GLM-5.3-Flash

GLM baseline 是 GLM 三个条件中表现最好的：TP=29、FN=4、FP=0、TN=33，严格准确率 93.94%，sensitivity 87.88%，specificity 100.00%。加入 v3 后，TP 降至 26、FN 增至 7，准确率降至 89.39%；加入 v4 后，TP 进一步降至 22、FN 增至 11，准确率降至 83.33%。

三个 GLM 条件的 FP 始终为 0，specificity 始终为 100.00%。因此 Skill 没有为 GLM 提供可继续改善的 test FP 空间，性能下降主要来自 FN 4→7→11，对应 sensitivity 87.88%→78.79%→66.67%。从输出行为看，额外 Skill 相当于让 GLM 更难作出阳性判断；这是一种观察到的 decision behavior，不代表已经知道模型内部如何实现决策。

### 6.2 DeepSeek-V4-Flash-Vision-Exp

DeepSeek baseline 的 TP 很高（32），FN 仅 1，sensitivity 达 96.97%；但 33 张阴性中有 18 张被判为阳性，specificity 仅 45.45%，严格准确率为 71.21%。其主要问题不是漏掉阳性，而是过多输出阳性。

Skill v3 将 FP 从 18 降到 11、TN 从 15 增到 22，严格准确率提高到 78.79%，specificity 提高到 66.67%；代价是 FN 从 1 增到 3，sensitivity 降到 90.91%。Skill v4 的 TP/FN 与 v3 相同，但 FP 进一步降到 9、TN 增到 24，严格准确率为 81.82%，specificity 为 72.73%。

因此，在 DeepSeek 上，Skill 的主要收益来自减少过度阳性预测，即 FP 18→11→9；代价是 FN 1→3→3。它表现为以一定 sensitivity 损失换取更明显的 specificity 和 overall accuracy 提升，而不是所有指标同时提高。

## 7. Skill 到底改变了什么

只比较 Accuracy 会掩盖两种模型完全不同的错误结构。下表中的 Δ 表示“后一个条件减去前一个条件”；paired counts 则直接统计同一张图在两个条件间的正确性变化。

| Model | Comparison | ΔTP | ΔFN | ΔFP | ΔTN | wrong→correct | correct→wrong | FP→TN | TN→FP | FN→TP | TP→FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM | baseline→v3 | −3 | +3 | 0 | 0 | 3 | 6 | 0 | 0 | 3 | 6 |
| GLM | baseline→v4 | −7 | +7 | 0 | 0 | 1 | 8 | 0 | 0 | 1 | 8 |
| GLM | v3→v4 | −4 | +4 | 0 | 0 | 3 | 7 | 0 | 0 | 3 | 7 |
| DeepSeek | baseline→v3 | −2 | +2 | −7 | +7 | 8 | 3 | 8 | 1 | 0 | 2 |
| DeepSeek | baseline→v4 | −2 | +2 | −9 | +9 | 10 | 3 | 10 | 1 | 0 | 2 |
| DeepSeek | v3→v4 | 0 | 0 | −2 | +2 | 6 | 4 | 5 | 3 | 1 | 1 |

对 GLM，可直接概括为：Skill 没有改变已经为 0 的 FP，却让更多真实阳性转成阴性。baseline→v3 有 3 个 wrong→correct，但有 6 个 correct→wrong；baseline→v4 只有 1 个 wrong→correct，却有 8 个 correct→wrong。所有这些 paired 改变都发生在阳性方向，即 FN→TP 与 TP→FN 的相互转换，净结果是 TP 减少、FN 增加。

对 DeepSeek，主要变化发生在阴性样本。baseline→v3 有 8 个 wrong→correct，全部为 FP→TN；同时出现 1 个 TN→FP 和 2 个 TP→FN。baseline→v4 有 10 个 FP→TN，同时有 1 个 TN→FP 和 2 个 TP→FN。因此，v3/v4 都明显减少了 baseline 的阳性过判，但这种修正不是无代价的，它也把少量原本正确的阳性改成了阴性。

v3→v4 的 DeepSeek 净混淆矩阵只变化为 FP 减 2、TN 加 2，但逐样本内部仍有双向交换：5 个 FP→TN、3 个 TN→FP、1 个 FN→TP、1 个 TP→FN。这说明同一个总计变化可能包含多组互相抵消的个体变化，不能把 v4 简化为“只修正了两张图”。六组 test 结果均没有 abstention 或 parse-failure transition。

用于口头汇报时，可以将两种行为概括为：**GLM 本来已经没有 false positive，Skill 主要增加 false negative；DeepSeek baseline 有大量 false positive，Skill 主要减少 false positive，但同时小幅增加 false negative。**

## 8. 图表解释

### 8.1 Figure 1：总体性能

![Figure 1：Held-out test 的 Accuracy、Sensitivity 与 Specificity](../experiments/final_test_analysis/figures/figure1_test_performance.png)

Figure 1 将六个条件的 Strict Accuracy、Strict Sensitivity 和 Specificity 放在同一张图中。GLM 三组的 specificity 都是 100%，但 Skill 越复杂，sensitivity 越低；DeepSeek baseline 的 sensitivity 很高而 specificity 很低，Skill v3/v4 则降低 sensitivity、提高 specificity，并使整体准确率上升。图中最重要的不是某一根柱子的绝对高度，而是两个模型对 Skill 呈现不同方向的权衡。

### 8.2 Figure 2：错误数量

![Figure 2：Held-out test 的 FP 与 FN 数量](../experiments/final_test_analysis/figures/figure2_test_error_counts.png)

Figure 2 是最直接的行为对比。GLM 的 FP 始终为 0，而 FN 从 4 增至 7、再增至 11；DeepSeek 的 FP 从 18 降至 11、再降至 9，而 FN 从 1 增至 3 后保持为 3。相比只看 Accuracy，这张图更清楚地显示 Skill 对两个模型产生了不同的 error trade-off。

### 8.3 Figure 3：DeepSeek 阴性来源分层

![Figure 3：DeepSeek 在不同阴性来源类别上的 FP rate](../experiments/final_test_analysis/figures/figure3_deepseek_negative_subtype_fp_rate.png)

Figure 3 按 Normal、Benign、Malignant 展示 DeepSeek 的 test FP rate。baseline 在三类中都出现 FP，因此其高 FP 并非完全由某一个来源类别造成。v4 在三类中的 FP rate 都低于 baseline；v3 在 Normal 和 Malignant 上下降，但 Benign 上从 44.44% 上升至 55.56%。这些结果只描述当前小样本中的输出行为，不能据此声称模型已经获得特定疾病的鉴别能力。

## 9. Paired comparison 与 McNemar exact test

由于各条件使用相同的 66 张 test 图像，可以用 exact McNemar test 对成对正确性进行补充比较。检验只使用不一致配对：`b = correct→wrong`，`c = wrong→correct`。

| Model | Comparison | correct→wrong (b) | wrong→correct (c) | Exact two-sided p |
|---|---|---:|---:|---:|
| GLM | baseline vs v3 | 6 | 3 | 0.507812 |
| GLM | baseline vs v4 | 8 | 1 | 0.039062 |
| GLM | v3 vs v4 | 7 | 3 | 0.343750 |
| DeepSeek | baseline vs v3 | 3 | 8 | 0.226562 |
| DeepSeek | baseline vs v4 | 3 | 10 | 0.092285 |
| DeepSeek | v3 vs v4 | 4 | 6 | 0.753906 |

GLM baseline vs v4 的 p=0.039062，为两个条件的 paired correctness 存在差异提供了统计证据；结合混淆矩阵和方向计数，这个差异主要表现为 v4 的 sensitivity 下降，而不是性能提升。

DeepSeek baseline vs v3 和 baseline vs v4 的 observed accuracy 与 FP 均改善，但 p 分别为 0.226562 和 0.092285，在 66 张 test 上均未低于传统的 0.05 阈值。因此，报告不能声称 DeepSeek Skill 获得了统计显著的性能提升。同时，p 值只反映当前配对样本下的统计证据，不代表临床重要性；p 大于 0.05 也不等同于两个条件完全相同。

## 10. Negative subtype analysis

33 张 test 阴性样本由 Normal 5 张、Benign 9 张、Malignant 19 张组成。由于各类别样本数不同，既要看 FP 数量，也要看 FP rate。

| Negative source class | N | DeepSeek baseline FP / rate | DeepSeek v3 FP / rate | DeepSeek v4 FP / rate |
|---|---:|---:|---:|---:|
| Normal | 5 | 2 / 40.00% | 1 / 20.00% | 1 / 20.00% |
| Benign | 9 | 4 / 44.44% | 5 / 55.56% | 3 / 33.33% |
| Malignant | 19 | 12 / 63.16% | 5 / 26.32% | 5 / 26.32% |

GLM 在 baseline、v3、v4 三个条件下，对 Normal、Benign、Malignant 的 FP 都为 0。DeepSeek baseline 的 18 个 FP 分布在三类中，其中 Malignant 的绝对 FP 最多（12），同时 FP rate 也最高（63.16%）；这不只是因为 Malignant 样本更多。

DeepSeek v3 的总 FP 从 18 降至 11，但改善并非三个 subtype 完全一致：Normal 为 2→1，Malignant 为 12→5，而 Benign 为 4→5。v4 相比 baseline 在三类中都观察到 FP 减少：Normal 2→1、Benign 4→3、Malignant 12→5。由于 Normal 只有 5 张、Benign 只有 9 张，这些比例对单个样本非常敏感，只应作为描述性分层结果，不应附加未经实验支持的医学解释。

## 11. 讨论

### 11.1 Skill 不是普遍增益

GLM baseline 已经具有 93.94% 的 test 严格准确率和 100.00% specificity，主要错误是 4 个 FN。v3/v4 没有改善 FP，却继续增加 FN，因而降低 sensitivity 和整体准确率。相反，DeepSeek baseline 的主要问题是 18 个 FP，v3/v4 正好改变了这类输出倾向，所以在牺牲少量 sensitivity 的同时提高了 specificity 和 observed accuracy。

这组结果支持一个有限但重要的结论：**Skill effectiveness is model-dependent。** 同一 Skill 不能脱离基础模型的原始错误结构来评价。对一种模型有用的“更谨慎”，在另一种本来就谨慎的模型上可能成为额外负担。

### 11.2 Skill 更像改变 decision behavior

本实验不支持把 Skill 简单理解为“加入领域知识后模型能力一定提高”。从 paired transitions 看，Skill 更直接地改变了模型如何组织可见证据、如何处理不确定性，以及何时愿意输出阳性。

对 DeepSeek，v3 后怀疑性阳性表述减少，更多阴性样本从 FP 转为 TN；同一种保守方向也把部分 TP 转为 FN。对 GLM，额外限制在 test 上几乎只表现为阳性判定减少。由于这里只能观察输入、输出和文本 findings，这些描述应被理解为外部 decision behavior，不能延伸为对模型内部推理机制的断言。

### 11.3 显式 checklist 的问题

GLM dev 的 v1/v2 结果表明，长篇疾病特征 checklist 可能让模型过度关注“某些典型表现没有出现”，并将这种缺失当作 negative evidence。v2 已明确说明“不应因为一个特征不可见就排除”，但整体 sensitivity 仍未恢复；删除清单的 v3 才回到 baseline 水平。

这是本项目在一个模型、一个小型 dev 集上的实验观察，不应写成关于所有模型或所有医学提示的普遍规律。它更适合作为 Skill 设计提醒：增加更多特征文字不一定增加有效信息，也可能改变模型的决策门槛。

### 11.4 Worked reasoning example 的作用不一致

v4 测试的是 worked example 能否在结构化说明之外进一步示范证据整合，而不是让模型记住一个具体形态模式。held-out test 上，v4 相比 v3 使 GLM 准确率从 89.39% 降至 83.33%，但使 DeepSeek 从 78.79% 升至 81.82%。

因此，worked example 没有跨模型一致效果，也不能据此声称 example 一定优于 instruction。对 GLM，它延续了更保守的阳性决策；对 DeepSeek，它又减少了 2 个净 FP，而没有进一步增加净 FN。这个差异进一步说明，提示形式的效果依赖模型原有的决策行为。

### 11.5 Dev 观察与 held-out test 是否复现

GLM v3 在 dev100 回到 baseline 严格准确率，但 held-out test 上低于 baseline 4.55 个百分点，说明开发集上的“恢复”没有完全转化为 test 提升。GLM v4 在公平 dev99 和 test 上都低于相应 baseline/v3，方向较一致。

DeepSeek v3/v4 在 dev 和 held-out test 上均表现出 FP 减少、specificity 提高、sensitivity 下降的方向。v4 test strict accuracy 相比 baseline 从 71.21% 提高到 81.82%，但 exact McNemar p=0.092285，不能写成统计显著提升。换言之，DeepSeek 的行为方向得到了 test 支持，但小样本下统计证据仍有限。

## 12. 局限性

1. 整个 benchmark 只有 166 张图像，held-out test 只有 66 张，指标容易受少量样本影响。
2. positive 与 negative 来自不同公开数据源，标签和来源可能同时相关。
3. 因此存在 dataset-source / domain bias 风险，模型可能利用与疾病无关的来源差异。
4. 不同医院、设备、图像标注样式和采集方式都可能形成 shortcut，本项目无法完全分离这些因素。
5. negative 中 Normal、Benign、Malignant 的构成来自现有数据集和比例抽样，不代表真实临床 prevalence。
6. 输入是单张静态 ultrasound 图像，不能替代包含多切面、实时扫查和动态征象的完整超声检查。
7. 数据缺少完整临床 metadata，无法结合病史、实验室检查、病灶位置和随访信息进行综合判断。
8. 本项目没有安排独立医生对全部图像和模型预测进行重新标注或盲法复核。
9. `confidence` 是模型 self-reported confidence，不是 calibrated probability，不能直接解释为真实正确概率。
10. DeepSeek 使用的是 Vision-Exp 模型，模型版本本身带有实验性质，未来服务版本变化可能影响复现。
11. 本实验属于 exploratory / pilot benchmark，Skill 迭代也主要依赖单个 dev split，存在开发集过拟合风险。
12. 这些结果不应被用于推断真实临床诊断性能，更不能作为患者诊断或治疗决策依据。

此外，McNemar p 值只补充说明成对预测是否出现不对称变化，并不衡量临床意义；阴性 subtype 的样本数也很小，尤其 Normal（n=5）和 Benign（n=9），分层比例的不确定性较大。

## 13. 结论

在这个 pilot benchmark 中，GLM baseline 本身表现较强，额外 Skill 没有带来提升。GLM 的 test FP 在 baseline、v3、v4 中始终为 0，Skill 的主要影响是 FN 从 4 增加到 7 和 11，使 sensitivity 和严格准确率持续下降。

DeepSeek baseline 则存在大量 false positive。Skill v3/v4 在 dev 和 held-out test 上都表现出减少 FP、提高 specificity 和 observed overall accuracy 的方向；test FP 从 18 降至 11 和 9，代价是 FN 从 1 增至 3。DeepSeek v4 的 test strict accuracy 为 81.82%，高于 baseline 的 71.21%，但当前 66 张 test 的 paired exact p 值未低于 0.05，不能夸大为统计显著或临床有效。

综合来看，Skill 的效果明显依赖基础模型原有的决策行为。领域 Skill 的设计不能简单等同于“加入更多医学知识”：显式 checklist、最小结构化流程和 worked reasoning example 都可能改变 evidence integration 与最终决策门槛，而且这种影响在不同模型上方向不一。对后续工作而言，更合理的路线是先识别基础模型的错误结构，再用独立数据验证 Skill 是否改善了目标错误，而不是不断在 test 结果上追加规则。

本项目到此停止 Skill 开发。所有实验方案、预测结果、prompt、Skill、backend、parser 与数据划分均保持冻结，不根据 held-out test 继续调参。
