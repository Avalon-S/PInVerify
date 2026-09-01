# Agent configs

Each YAML here is one full agent specification: prompt files, verification mode,
fusion rule, NBV policy, step budget, dataset paths, and server endpoints.
Run one with `scripts/evaluate.py --config configs/agent/<name>.yaml`, or a whole
set with `run_all.sh` (see the reproduction map below).

## Naming

```
[backbone_]<view mode>_<query mode>[_adaptive]_<nbv>[_merged].yaml
   |            |          |            |         |        |
   |            |          |            |         |        └ description fed as one merged sentence
   |            |          |            |         └ next-best-view policy: random | fps | llm
   |            |          |            └ adaptive stopping (attr-majority fusion, up to T=6 steps)
   |            |          └ attr (attribute decomposition) | direct (holistic) | merged
   |            └ single_view (1 step) | multi_view
   └ none = Qwen3-VL MLLM; clip_ / siglip2_ = embedding-only; trained_ = LoRA fine-tuned
```

Everything the paper reports uses **adaptive stopping** over a 6-sector topology.
Every config here maps to a row of a paper table; earlier fixed-budget variants
that no table uses are not published.

One exception: `nbv: viewhint` is implemented in `pver/policies/nbv.py` and its
prompts ship in `configs/prompts/`, but no paper table uses it, so no config here
selects it.

## Paper reproduction map

All numbers below are the 3,000-episode test split (`pv_index_all.jsonl`),
Grounding DINO detection, Qwen3-VL-4B unless a different backbone is named.
Overall / Pos / ASD are quoted from the paper so you can check a run end to end.

### Table 4: training-free main table

`AGENT_SET=main bash run_all.sh`

| Paper row | Config | Ovr | Pos | ASD |
|---|---|---|---|---|
| Single-View / Attr | `single_view_attr` | 0.844 | 0.652 | 1.0 |
| Single-View / Direct | `single_view_direct` | 0.813 | 0.457 | 1.0 |
| Single-View / Merged | `single_view_merged` | 0.815 | 0.468 | 1.0 |
| MV attr + Random | `multi_view_attr_adaptive_random` | 0.849 | 0.592 | 2.34 |
| MV attr + Angular FPS | `multi_view_attr_adaptive_fps` | 0.848 | 0.592 | 2.36 |
| MV attr + LLM-NBV (TF-Best) | `multi_view_attr_adaptive_llm` | 0.850 | 0.596 | 2.34 |
| MV direct + Random | `multi_view_direct_adaptive_random` | 0.827 | 0.492 | 2.09 |
| MV direct + Angular FPS | `multi_view_direct_adaptive_fps` | 0.826 | 0.490 | 2.13 |
| MV direct + LLM-NBV | `multi_view_direct_adaptive_llm` | 0.825 | 0.486 | 2.08 |

### Table 5: cross-model comparison

The MLLM rows reuse the Table 4 configs against a different backbone server;
only the server launcher changes, not the YAML.

| Paper row | Config | Server | Ovr |
|---|---|---|---|
| CLIP / SV-Merged | `clip_single_view_merged` | `start_multigpu_servers.sh` | 0.771 |
| SigLIP2 / SV-Merged | `siglip2_single_view_merged` | `start_multigpu_servers.sh` | 0.801 |
| Qwen3-VL-4B / MV-Attr+LLM | `multi_view_attr_adaptive_llm` | `start_multigpu_servers.sh` | 0.850 |
| Qwen3-VL-8B / MV-Attr+LLM | `multi_view_attr_adaptive_llm` | same, `--model` pointed at Qwen3-VL-8B | 0.797 |
| SenseNova-SI / MV-Direct+Rnd | `multi_view_direct_adaptive_random` | `start_multigpu_servers_sensenova.sh` | 0.833 |

The embedding baselines load CLIP/SigLIP2 weights in-process from
`method.clip_model` (`./models/clip-vit-large-patch14`,
`./models/siglip2-so400m-patch14-384`). They still need the Qwen text server for
description merging and GDINO for detection, so the same launcher applies.

`AGENT_SET=embedding bash run_all.sh` runs the two CLIP/SigLIP2 rows plus the
appendix sweep (Table 18), which covers their multi-view variants.

### Table 6: trained agents

Trained agents do not use these YAMLs directly. They all run through
`configs/agent/trained_e2e.yaml` plus a LoRA adapter served by
`scripts/start_multigpu_servers_lora.sh`, driven by `scripts/eval_trained.sh`:

| Paper row | Adapter | Det. | Ovr | Pos |
|---|---|---|---|---|
| TF-Best | none (`multi_view_attr_adaptive_llm`) | DINO | 0.850 | 0.596 |
| Base (no FT) | none | DINO | 0.706 | 0.146 |
| SFT | `training/run_sft.sh` (Generic-CoT, v2 data) | DINO | 0.848 | 0.759 |
| SFT + GRPO | `training/run_grpo.sh` | DINO | 0.853 | 0.736 |
| SFT + GSPO | `training/run_gspo.sh` | DINO | 0.856 | 0.745 |
| SFT + GSPO | `training/run_gspo.sh` | GT | 0.889 | 0.813 |

The `*_v3.sh` training scripts are the Specific-CoT variant reported in
Appendix F (Table 23), not the main table. See [docs/TRAINING.md](../../docs/TRAINING.md).

`trained_sft.yaml` / `trained_grpo.yaml` / `trained_gspo.yaml` are convenience
wrappers around the same agent with a fixed output directory; the paper runs
used `trained_e2e.yaml`.

### GT-box ablations (Sec. 6.4)

Rerun any config above with `method.bbox_mode=gt`, or `BBOX_MODES=gt bash run_all.sh`.
Reference points: `single_view_attr` 0.875 (Pos 0.758), `single_view_direct` 0.823,
`multi_view_attr_adaptive_llm` 0.884 (Pos 0.728).

## Paths

Configs assume the dataset is at `./data/pv_dataset` and backbone weights at
`./models/<name>`. Override per run without editing the file:

```bash
python scripts/evaluate.py --config configs/agent/multi_view_attr_adaptive_llm.yaml \
    dataset.root=/mnt/data/pv_dataset \
    dataset.index_file=pv_index_50.jsonl \
    method.bbox_mode=gt
```
