# Architecture

PInVerify is structured as a thin agent loop on top of a deterministic, offline navigation environment. This page walks through the codebase top-down so new contributors can find their footing.

## High-level flow

```
                    +-------------------+
                    |   Episode (val)   |
                    |   meta.json +     |
                    |   6-sector views  |
                    +---------+---------+
                              |
                              v
       +----------------------+----------------------+
       |              PInVerifyEnv (pver/envs)       |
       |  - sector graph, navigability, trap views   |
       |  - returns (rgb, depth, mask, gt_bbox)      |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |          Agent Policy (pver/policies)       |
       |   MLLM | CLIP | trained_e2e | random        |
       |        +-------------+-----------+          |
       |        |             |           |          |
       |  detector       attribute    next-best-view |
       |  (DINO/GT)    verifier     selector (NBV)   |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |        Tracker / Fusion (pver/policies)     |
       |  per-attribute Matched/Contradictory/Missing|
       |  -> Decision (YES / NO / continue)          |
       +----------------------+----------------------+
                              |
                              v
                    +---------+---------+
                    |   Metrics (eval)  |
                    |  Overall, Pos,    |
                    |  NegSame, NegDiff |
                    |  ASD, NavFail     |
                    +-------------------+
```

## Package tour

### `pver/envs/env.py`

`PInVerifyEnv` is the offline simulator. It loads the captured 6-sector views, exposes:

- `reset(episode_id)` — initial observation at the agent's starting sector
- `step(action)` — `NAV(direction)`, `STOP_YES`, `STOP_NO`
- `current_observation()` — `{rgb, depth, mask, gt_bbox, sector_id, navigable_mask, visibility_warning}`

The environment has zero physics — it just looks up pre-rendered captures. Direction → sector resolution uses angular positions of all available captures relative to the goal, *not* sequential sector IDs (see [memory note](#design-traps)).

### `pver/policies/`

Pluggable agents that consume an observation stream and emit an action stream.

| Module | Role |
|---|---|
| `mllm_policy.py` | Reference training-free agent: detection (DINO/GT) → category routing → attribute extraction → per-view verification → fusion |
| `clip_policy.py` | Embedding-only baseline (CLIP / SigLIP-2) |
| `trained_policy.py` | LoRA-fine-tuned end-to-end Qwen3-VL agent (SFT / DPO / GRPO / GSPO) |
| `random_policy.py` | Sanity baseline |
| `nbv.py` | NBV strategies: `RandomNBV`, `LLMBasedNBV`. FPS-NBV lives inline in MLLMPolicy |
| `tracker.py` | `AttributeStateTracker` accumulates per-view evidence across viewpoints |
| `fusion.py` | Decision modules: visibility-weighted, asymmetric-threshold, majority-vote, veto, LLM-based |
| `server_client.py` | HTTP wrappers around VLM / detector / LLM endpoints |
| `server_client_pool.py` | Connection pool across multiple GPU servers |

### `pver/eval/metrics.py`

`calculate_metrics()` computes:

- `Overall` — accuracy across all 3,000 episodes
- `Pos / NegSame / NegDiff` — per-pair-type accuracy
- `ASD` (Average Steps to Decision)
- `NavFail` — fraction of episodes that hit step budget or unreachable sector
- First-view accuracy, per-category breakdown, ASD distribution

### `pver/data/builder.py`

`IndexBuilder` reads PInNED's flat episode export and emits the 1:1:1 pair-stratified jsonl indices used by the evaluator. `NegativeSampler` does the same-/diff-category pairing.

### `pver/utils/`

| Module | Role |
|---|---|
| `prompt_loader.py` | Resolve `${...}` placeholders in YAML prompt templates at runtime |
| `visualizer.py` | Per-episode HTML dump for qualitative inspection |
| `sector_viz.py` | Render the 6-sector topology figure (used in the paper) |

## Server architecture

Heavy models run as standalone HTTP services so the evaluator can scale across GPUs without holding model state itself.

```
┌──────────────────────────────┐    HTTP    ┌────────────────────┐
│ scripts/evaluate_multigpu_…  │ ─────────▶│ Qwen3-VL server     │
│ (orchestrator, no GPU)       │           │ port 12182..12482   │
│                              │ ─────────▶│ DINO / SenseNova    │
└──────────────────────────────┘           └────────────────────┘
```

Servers are launched via shell scripts in `scripts/start_multigpu_servers*.sh`. The evaluator reaches them via `pver/policies/server_client.py`. Each server exposes the same minimal API:

- `POST /qwen-vl` — multi-image VLM query, returns text
- `POST /qwen-text` — text-only LLM query (no image)
- `POST /gdino` — detection on an image+text query
- `GET  /health` — liveness check

## Config system

PInVerify uses Hydra/OmegaConf. Every entrypoint takes a `--config` and accepts CLI overrides:

```bash
python scripts/evaluate.py --config configs/agent/multi_view_attr_llm.yaml \
  dataset.root=./data/pv_dataset \
  method.bbox_mode=dino \
  +start_idx=0 +end_idx=500
```

Defaults reference `./data/pv_dataset` and `./outputs/` — override these or set `PV_DATA_ROOT` if you prefer.

## Design traps

A few things that have bitten contributors before:

- **Sector IDs are not angular**. Don't compute directional navigation as `(cur_sector + offset) % 6`. Compute actual angular positions of captures relative to the goal and resolve targets within ±30°.
- **GT-mode trap views need full-image fallback**. When `method.bbox_mode=gt` and the GT mask doesn't meet the visibility threshold, use the full image with confidence 0.1 — *don't* fall back to DINO (that breaks the controlled-detection comparison).
- **JSONL index vs. meta.json**. The episode JSONL has `target_object_id` + `query_object_id`. `meta.json` has only `object_id` (= target). All policies should prefer the JSONL fields; never let `meta.json` overwrite them.
- **Coordinate system**. Habitat uses `-Z = forward`. The top-down visualizer negates Z (`goal_z = -goal_pos[2]`).
