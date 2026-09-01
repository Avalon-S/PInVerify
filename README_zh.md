# PInVerify：面向主动实例验证的离线具身基准

[English](README.md) · **简体中文**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2605.30639-red)](https://arxiv.org/abs/2605.30639)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://avalon-s.github.io/PInVerify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FMEA @ CVPR 2026](https://img.shields.io/badge/FMEA-CVPR%202026-blueviolet)](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/)

FMEA @ CVPR 2026 论文 **"PInVerify: An Offline Embodied Benchmark for Active Instance Verification"** 的官方仓库。

导航能把 agent 带到正确的**类别**跟前。但分辨眼前两个背包哪个才是要找的那个，是另一回事，也正是这个基准单独拎出来的问题：agent 已经站在候选物体旁边，接下来要靠选择从哪个角度看，判断这到底是不是描述里说的那一个。

这个任务叫**主动实例验证（Active Instance Verification, AIV）**。基准包含 3000 个 episode，视点按 6 扇区拓扑组织，并附带免训练和 LoRA 微调两类参考 agent。

<p align="center">
  <img src="project-page/static/images/overview.png" width="92%" alt="主动实例验证：agent到达候选物体附近后，需要选择视角判断它是否与描述相符">
</p>

## 任务设定

每个 episode 把目标放在一圈预渲染的视点中间：6 个角度扇区，各有远近两环。agent 每步选择下一个视角，最多 6 步。有两种失败与识别能力无关：某个扇区可能根本走不到，或者能走到但没用，因为从那里几乎看不见目标。基准把这两种都记为导航失败，而不是把它们藏起来。

<p align="center">
  <img src="project-page/static/images/sector_topology.png" width="82%" alt="目标周围的 6 个角度扇区，分远近两环，标注了trap view和不可达扇区">
</p>

参考 agent 是按属性逐条工作的。它先把描述拆开，每个属性拿到当前视角下核对一遍，再用一个 tracker 把不同视角之间相互矛盾的证据合并起来。一旦 tracker 收敛就停下，不会把剩下的步数用完。

<p align="center">
  <img src="project-page/static/images/method_pipeline.png" width="92%" alt="流程：属性拆解、逐视角验证、tracker 合并、下一最佳视角选择、自适应停止">
</p>

## 三个分支

项目分三个分支，各自的 Python 环境**互不兼容**，这也正是它们分成分支而不是目录的原因：一个 checkout 对应一个 conda 环境，不会装错。

| 分支 | 内容 | 环境 |
|---|---|---|
| **`main`**（当前分支）| 基准本体：环境、免训练与微调 agent、评测框架、论文每张表对应的配置 | Python 3.10, torch 2.1.2+cu118, transformers 4.52.4, numpy 1.26 |
| **`data-collection`** | 从 PInNED 构建本基准的采集流水线，对应论文附录 A：物体注入、6 扇区视点采集、mask 与可达性标注、迭代修复循环，以及附录 A.10 的渲染审计 | Python 3.9, torch 1.13.1+cu117, habitat 0.2.3, pytorch3d, transformers 4.38.2, numpy 1.23 |
| **`benchmark-fix`** | 构建数据集过程中发现的上游 PIN 基准缺陷及其检测修复代码：物体穿模、地面高度错误、异常 episode 分类、goal-view 评分。**不属于论文内容**，请不要去论文里找对应章节 | 与 `data-collection` 同一套 habitat 环境 |

冲突是硬性的：torch 1.13 与 2.1 分属不同 CUDA build（cu117 与 cu118），numpy 1.23 与 1.26，transformers 4.38 与 4.52（Qwen3-VL 需要新版）。每个分支保持独立的 conda 环境。

微调代码在 `main` 的 [`training/`](training/) 下，和它训练的那些 agent 放在一起，运行在 `main` 环境加 `ms-swift`。

```bash
git clone https://github.com/Avalon-S/PInVerify.git
cd PInVerify
git checkout data-collection    # 或 benchmark-fix
```

`benchmark-fix` 是副产品而非论文声称的贡献。它只发布检测与修复代码，不发布修复后的数据：PIN 原始数据是公开的，流水线可以针对它重新运行。

## `main` 的发布进度

基准分两个阶段发布，这样免训练那一半可以先用起来。

### 阶段 1：免训练基线与测试集

| 内容 | 状态 |
|---|---|
| 环境、tracker、融合、NBV 策略（`pver/`）| 已发布 |
| 免训练 agent 配置（表 4 与表 5）| 已发布，映射见 [configs/agent/README.md](configs/agent/README.md) |
| 评测框架与多卡运行器 | 已发布 |
| VLM 与检测器服务 | 已发布 |
| 3000 episode 测试集、索引、缓存 | 尚未上传 |
| 绘图脚本 | 已发布 |

### 阶段 2：微调基线与训练数据

| 内容 | 状态 |
|---|---|
| SFT / DPO / GRPO / GSPO 的数据准备与训练脚本（`training/`）| 已发布 |
| 微调 agent 配置与评测入口 | 已发布 |
| SFT 与 RL 训练池（各 15,225 对，含 crop）| 尚未上传 |
| LoRA 权重，8 个变体 | 尚未上传 |
| 训练曲线 | 尚未上传 |

所有标记"尚未上传"的内容，都通过 [`scripts/prepare_hf_release.py`](scripts/prepare_hf_release.py) 从采集机器推送到 Hugging Face。对应的代码路径都已经在仓库里，也都有文档。

代码按产出论文数字时的原样发布，没有在干净环境里重跑过端到端流程，所以路径和服务端口需要按你自己的环境调整。

## 目录结构

```
PInVerify/
├── pver/                 核心包：环境、策略、tracker、融合、NBV、评测、可视化
├── configs/
│   ├── agent/            25 个 agent 配置 + 论文表格到配置的映射 README
│   └── prompts/          10 个 prompt 模板（拆解 / 验证 / 类别 / 融合 / 导航）
├── scripts/              评测入口、缓存构建、绘图脚本、HF 发布工具
├── training/             SFT + DPO/GRPO/GSPO 的数据准备与训练脚本
├── servers/              VLM 与检测器服务（Qwen3-VL、Grounding DINO、CLIP、SenseNova-SI）
├── data/examples/        用于自检的少量样例 episode
├── docs/                 INSTALL / DATASET / EVALUATION / TRAINING / ARCHITECTURE
├── hf_cards/             Hugging Face 数据集与模型卡片，以及上传流程
├── project-page/         静态项目页（自动部署到 GitHub Pages）
├── run_all.sh            免训练评测运行器（复现表 4 与表 5）
└── runner.py             批量评测框架
```

配置中的路径默认数据集在 `./data/pv_dataset`，模型权重在 `./models/<name>`，都可以在命令行按次覆盖。

## 快速开始

### 1. 安装

```bash
git clone https://github.com/Avalon-S/PInVerify.git
cd PInVerify

conda create -n pv_bench python=3.10 -y
conda activate pv_bench
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

这一步覆盖 Qwen3-VL agent 和 embedding 基线。另外三样需要单独安装，**完整步骤见 [docs/INSTALL.md](docs/INSTALL.md)**，其中包括第一样最容易踩的 CUDA 版本约束：

- **Grounding DINO**：`method.bbox_mode=dino`（论文主表用的模式）需要它。它会编译 CUDA 扩展，版本必须与所在环境的 torch CUDA build 一致，`servers/run_groundingdino_server.py` 要在它的 checkout 里运行。
- **SenseNova-SI**：只用于表 5 的一行。它需要独立的 `sensenova` 环境（Python 3.11、torch 2.5.1+cu121、numpy 2.x），无法与 `pv_bench` 共存。
- **ms-swift**：只在重新训练时需要。它要求的 transformers 比 `requirements.txt` 里钉的版本更新，所以放在单独的 `pv_train` 环境（torch 2.2.0、transformers 4.57.0）。训练出的 adapter 仍然在 `pv_bench` 里评测。

### 2. 下载数据集

> 阶段 1 的上传尚未公开。数据集在采集机器上组装并从那里推送，仓库公开后这里会给出下载命令。

发布后的目录结构：

```
data/pv_dataset/
├── pin_capture/
│   ├── val/<scene>/<episode>/{meta.json, rgb/, mask/, overview.png}
│   ├── train_sft/...
│   └── train_rl/...
├── image_gt/<category>/       GT 包围框 mask
├── val/pv_index_{50,100,500,1000,all}.jsonl
├── train_sft/{pv_train_sft_index.jsonl, sft_data_v2.jsonl, sft_data_v3.jsonl, crops/, crops_v3/}
├── train_rl/{pv_train_rl_index.jsonl, rl_data_v2.jsonl, dpo_data_v3.jsonl, crops_rl/, crops_dpo/}
├── attr_cache.json
├── category_cache.json
├── merge_cache.json
└── object_descriptions_with_category.json
```

`pv_index_all.jsonl` 是论文所有数字使用的 3000 episode 测试集。完整规格见 [docs/DATASET.md](docs/DATASET.md)。

在持有数据的机器上发布（不要在拷贝上做）：

```bash
python scripts/prepare_hf_release.py check  --data-root <数据集根目录> --splits val
python scripts/prepare_hf_release.py upload --data-root <数据集根目录> --splits val \
    --repo Avalon-S/PInVerify --card hf_cards/dataset_card.md
```

`check` 会遍历每个 split 的索引，发现残缺就拒绝上传。

### 3.（可选）下载微调权重

论文报告的每个权重都在同一个模型仓库 `Avalon-S/PInVerify-Qwen3VL-4B` 里，按子目录组织。下表是 3000 episode 测试集上的总体准确率，括号内为正样本准确率：

| 权重 | DINO | GT |
|---|---|---|
| *（基座模型，未微调）* | 0.706 (0.146) | 0.710 (0.161) |
| `generic-cot/sft` | 0.848 (0.759) | 0.877 (0.828) |
| `generic-cot/grpo` | 0.853 (0.736) | 0.887 (0.806) |
| **`generic-cot/gspo`** | **0.856** (0.745) | **0.889** (0.813) |
| `specific-cot/sft` | 0.858 (0.697) | 0.884 (0.761) |
| `specific-cot/dpo-200` | 0.859 (0.700) | 0.881 (0.756) |
| `specific-cot/dpo-400` | 0.860 (0.665) | 0.884 (0.729) |
| `specific-cot/grpo` | 0.855 (0.793) | 0.884 (0.847) |
| `specific-cot/gspo` | 0.851 (0.796) | 0.889 (0.813) |

`generic-cot/*` 对应论文主表（表 6），`specific-cot/*` 是附录 F 的变体，两者只在 SFT 阶段 CoT 的写法上不同。`generic-cot/gspo` 是论文的最好结果。95% 置信区间约为 ±1.3 个百分点，所以微调各变体之间的差异大多在噪声范围内；真正拉开差距的是与未微调基座的对比，后者几乎总是回答"不匹配"。

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B \
    --include "generic-cot/gspo/*" --local-dir ./models/pinverify
```

尚未上传，仓库公开后本节生效。

### 4. 在 50 episode 小样本集上跑一个配置

```bash
# 单卡启动一个 Qwen3-VL-4B 服务。
# 模型路径是脚本顶部的 MODEL_PATH 常量；多卡启动器用的批处理服务则接受 --model 参数。
python servers/run_qwen3_server.py --port 12182

# 另开一个 shell：用 GT 框在 50 个 episode 上跑 MV-Attr+LLM-NBV
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_adaptive_llm.yaml \
  dataset.index_file=pv_index_50.jsonl \
  method.bbox_mode=gt
```

`method.bbox_mode=dino` 还需要在 12183 端口起一个 Grounding DINO 服务，从 GroundingDINO 的 checkout 里运行 `servers/run_groundingdino_server.py`，具体步骤见该文件的 docstring。

结果落在 `./outputs/<run_name>/metrics.json`。

## 复现论文结果

复现需要数据集，微调那张表还需要权重，所以在两者发布之前无法进行。下面是预期的入口命令。

哪个配置产出哪一行，连同预期精度，都记录在 **[configs/agent/README.md](configs/agent/README.md)** 里。

### 表 4：免训练主表

```bash
# 4 卡动态调度，跑 9 个免训练配置，DINO 检测
bash scripts/start_multigpu_servers.sh 4
bash run_all.sh                      # 完整 3000 episode
bash run_all.sh 50                   # 小样本集

BBOX_MODES=gt bash run_all.sh        # 6.4 节的 GT 框消融
```

汇总（`compare_metrics.py` 的目录是位置参数）：

```bash
python scripts/summarize_all_agents.py --output_base ./outputs/main_all
python scripts/compare_metrics.py ./outputs/main_all --breakdown
python scripts/compare_metrics.py ./outputs/main_all --csv table4.csv
```

时间上要有准备：8 张 RTX 3090 跑完主表这 9 个配置，实际耗时约 17 小时，折合 153 GPU 小时，其中最重的 MV-Attr+LLM 一个就占 31 小时。加上 `BBOX_MODES=gt` 大约翻倍。硬件建议见 [docs/INSTALL.md](docs/INSTALL.md)。

### 表 5：跨模型对比

```bash
# CLIP / SigLIP2 embedding 基线（含附录的扫描）
AGENT_SET=embedding bash run_all.sh

# Qwen3-VL-8B：同样的配置，服务指向 8B 权重
# SenseNova-SI: bash scripts/start_multigpu_servers_sensenova.sh 4
```

### 表 6：微调 agent

```bash
# 重新训练（用已发布权重则跳过）
bash training/run_sft.sh             # Generic-CoT SFT，对应论文主表
bash training/run_gspo.sh            # SFT + GSPO，最好结果

# 评测某个权重
ADAPTER=./outputs/training/gspo_v2_from_sft bash scripts/start_multigpu_servers_lora.sh 4
bash scripts/eval_trained.sh gspo_v2
```

`*_v3.sh` 训练的是附录 F 报告的 Specific-CoT 变体，区别见 [docs/TRAINING.md](docs/TRAINING.md)。

### 论文插图

绘图脚本从各自文件顶部硬编码的路径读取运行输出（`./outputs/qwen3_vl_4b/...` 和 `./outputs/trained/...`）。要么改这些常量，要么把运行结果产出到脚本期望的位置：

```bash
OUT_BASE=./outputs/qwen3_vl_4b SAVE_VIZ=true bash run_all.sh

python scripts/plot_per_category.py        # 分类别精度，读 metrics.json
python scripts/plot_nbv_polar_dino.py      # NBV 方向极坐标图，读 episode.json
python scripts/plot_case_study.py          # 单个定性案例，读 episode.json
python scripts/plot_dataset_overview.py    # 数据集统计，直接读数据集
```

`SAVE_VIZ=true` 会额外写出每个 episode 的 `episode.json`，后两个脚本需要它。这只占磁盘，不影响任何指标。

完整复现指南见 [docs/EVALUATION.md](docs/EVALUATION.md) 与 [docs/TRAINING.md](docs/TRAINING.md)。

## 引用

```bibtex
@misc{jiang2026pinverifyofflineembodiedbenchmark,
      title={PInVerify: An Offline Embodied Benchmark for Active Instance Verification},
      author={Yuhang Jiang},
      year={2026},
      eprint={2605.30639},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30639},
}
```

## 致谢

PInVerify 构建在 [PInNED (Barsellotti et al., NeurIPS 2024)](https://arxiv.org/abs/2410.18195) 之上，使用了它的 HM3D 场景、Objaverse-XL 物体池和实例描述。感谢作者公开这项工作。

## 许可

基于 [MIT License](LICENSE) 发布。基准数据沿用 PInNED 的条款，数据使用条件请查阅其许可。
