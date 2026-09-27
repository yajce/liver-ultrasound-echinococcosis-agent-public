# 实验报告：基于大模型 Agent 的肝脏超声包虫病识别

> ⚠️ 本研究为科研入门实验，模型输出不构成临床诊断依据。

## 1. 研究背景与目的

TODO: 包虫病背景、超声筛查意义、大模型多模态判读的动机。

## 2. 数据

TODO: 图像来源与数量、类别分布、标注方式（labels.csv）、伦理/使用许可说明。

## 3. 方法

TODO:
- 模型与后端（GLM-5.3-Flash vs 对比模型）
- 实验设置：无 Skill / 有 Skill（prompt_version 对应关系）
- Skill 判读流程概述（引用 skills/echo-hydatid/SKILL.md 及其文献）

## 4. 结果

TODO: 2 模型 × 2 设置的 Accuracy / Precision / Recall / F1 表格 + 混淆矩阵；
解析失败率、latency 分布；典型案例（含 raw_response 摘录）。

## 5. 讨论与局限性

TODO: 样本量、类别不平衡、单一模态、prompt 敏感性；误差案例分析。

## 6. 结论

TODO。
