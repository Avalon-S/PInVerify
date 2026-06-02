# Evaluation Guide

This page covers running the training-free evaluation matrix and reproducing Table 5 of the paper.

## Prerequisites

1. Dataset downloaded to `./data/pv_dataset/` ([DATASET.md](DATASET.md))
2. Qwen3-VL-4B (or 8B) weights at `./models/Qwen3-VL-4B-Instruct`
3. Optionally: Grounding DINO server running (for `method.bbox_mode=dino`)
4. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118`

## Quickest path — single configuration on a smoke split

```bash
# Start Qwen3-VL server (one GPU)
python servers/run_qwen3_server.py --port 12182 --model ./models/Qwen3-VL-4B-Instruct &

# Evaluate MV-Attr + LLM-NBV on 50 episodes
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_llm.yaml \
  +start_idx=0 +end_idx=50 \
  dataset.index_file=pv_index_sectors6_50.jsonl \
  method.bbox_mode=gt \
  output.root=./outputs/smoke
```

Result: `./outputs/smoke/multi_view_attr_llm_gt_50/metrics.json` with overall / Pos / Neg_Same / Neg_Diff / ASD.

## Multi-GPU full sweep (Table 5)

```bash
# Boot 4 servers (one per GPU), tmux'd
bash scripts/start_multigpu_servers.sh

# Wait for /health to return 200 on all four
for p in 12182 12282 12382 12482; do curl -fs http://127.0.0.1:$p/health; done

# Run the 18-config × {GT, DINO} sweep
bash run_all.sh 3000

# Tear down
bash scripts/manage_servers_multigpu.sh stop
```

`run_all.sh` calls `scripts/evaluate_multigpu_dynamic.py`, which schedules per-episode work across the 4 servers with dynamic load balancing.

## Agent config matrix (paper Table 5)

| Family | Query | Single-view | Multi-view variants |
|---|---|---|---|
| **Qwen3-VL TF** | Direct | `single_view_direct.yaml` | `multi_view_direct_{random,fps,llm}.yaml` |
| **Qwen3-VL TF** | Attr | `single_view_attr.yaml`   | `multi_view_attr_{random,fps,llm}.yaml`   |
| **LoRA TF**   | Direct | `lora_single_view_direct.yaml` | `lora_multi_view_direct_{random,fps}.yaml` |
| **CLIP**     | Embed  | `clip_single_view.yaml`    | `clip_multi_view_{random,fps}.yaml`        |
| **SigLIP-2** | Embed  | `siglip2_single_view.yaml` | `siglip2_multi_view_{random,fps}.yaml`     |
| **Trained**  | E2E    | —                          | `trained_{e2e,sft,grpo,gspo}.yaml`         |

All 21 configs live in [`configs/agent/`](../configs/agent/).

## Detection mode

| `method.bbox_mode` | Description |
|---|---|
| `gt`   | Use ground-truth bounding boxes (oracle upper bound) |
| `dino` | Use Grounding DINO (deployment-realistic; reported in Table 5 main) |

Toggle on the command line:

```bash
python scripts/evaluate.py --config <cfg> method.bbox_mode=dino
```

## Aggregation & figures

```bash
# Tabulate every metrics.json under outputs/
python scripts/summarize_all_agents.py --root ./outputs/sectors6_3000

# Pairwise comparison report
python scripts/compare_metrics.py --root ./outputs/sectors6_3000

# Paper-quality figures
python scripts/generate_report_figs.py
python scripts/plot_per_category.py
python scripts/plot_nbv_polar_dino.py
python scripts/plot_case_study.py
```

## Statistical notes

The paper reports 95% binomial CI ≈ ±1.3 pp at p=0.85 on n=3,000. Because episodes are not i.i.d. (71 unique instances × ~42 episodes each), differences below 2 pp should be treated as within cluster CI; we recommend instance-level paired bootstrap or McNemar's test for close comparisons.

## Smoke-test reference numbers (50 episodes)

| Config | Overall | Pos | Neg_Same | Neg_Diff |
|---|---|---|---|---|
| `single_view_attr` (GT)         | ~0.86 | ~0.65 | ~0.94 | ~0.98 |
| `multi_view_attr_llm` (GT)      | ~0.87 | ~0.62 | ~0.97 | ~0.99 |
| `multi_view_attr_random` (GT)   | ~0.86 | ~0.61 | ~0.96 | ~0.99 |

Numbers may vary ±5 pp due to small n; use for pipeline sanity, not for citation.
