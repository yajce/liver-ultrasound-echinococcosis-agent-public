# 肝脏超声包虫病识别 Agent（科研项目）

基于大模型的肝脏超声图像包虫病识别实验框架：输入超声图像，输出结构化识别结果。
支持"无 Skill / 有 Skill"两种实验设置，以及可替换的大模型后端（GLM / GPT / Claude）。

> ⚠️ **免责声明**：本项目为科研入门实验。所有模型输出仅为研究实验结果，
> **不构成临床诊断依据**，不得用于任何临床决策。

## 目录结构

```
echinococcosis-agent/
├── config.yaml               # 实验配置（不含任何密钥）
├── .env                      # API key（本地填写，不入库）
├── .env.example              # 环境变量模板
├── data/
│   ├── images/               # 所有超声图像统一放这里，平铺，不按类别建子目录
│   └── labels.csv            # 唯一的真实标签来源：image_id, true_label
├── skills/
│   └── echo-hydatid/SKILL.md # 包虫病判读流程（模板，内容待文献补充）
├── src/
│   ├── prompts.py            # baseline / Skill prompt 管理与版本号
│   ├── agent.py              # Agent：组装 prompt → 调用模型 → 生成记录
│   ├── parser.py             # 从原始回答解析结构化结果
│   ├── runner.py             # 单张 / 批量实验入口，写出 results.csv
│   ├── evaluate.py           # 离线指标计算
│   └── models/               # 可替换的大模型后端
│       ├── base.py           # ModelBackend 抽象接口
│       ├── glm_backend.py
│       ├── gpt_backend.py
│       └── claude_backend.py
├── experiments/              # 每次实验一个子文件夹（配置快照 + 结果 + 指标）
└── docs/
    └── report_template.md    # 实验报告框架
```

## 数据规范

- 图像统一放在 `data/images/`（**平铺，不建类别子目录**，避免目录名泄露标签）。
- 真实标签只保存在 `data/labels.csv`，两列：`image_id,true_label`。
  `image_id` 为图像文件名（含扩展名），`true_label` 的类别集合在确定数据后统一规定。

## 环境变量

复制 `.env.example` 为 `.env` 并填入 API key。`.env` 已被 `.gitignore` 忽略，
**任何情况下不要提交或把 key 写入其他文件**。

## 快速开始

离线 mock 链路（无需 API key，当前可用）：

```bash
python -m src.runner                                        # 单条模拟样本
python -m src.runner --batch                                # 按 labels.csv 批量（断点续跑）
python -m src.runner --batch --run-dir experiments/mock_batch_test
python -m src.evaluate experiments/mock_batch_test/results.csv
```

GLM 真实单图（需在 `.env` 配置 `THU_MAAS_*`，并 `pip install openai python-dotenv`）：

```bash
python -m src.runner --backend glm --image test.jpg
```

接入真实后端后（TODO）：

```bash
cp .env.example .env       # 填入 key
python -m src.runner --single data/images/xxx.jpg
python -m src.runner --batch
```
