# Hugging Face Hub cards

Cards and upload recipes for the two Hub repositories.

| File | Target repo |
|---|---|
| `dataset_card.md` | dataset: `Avalon-S/PInVerify` |
| `model_card.md` | model: `Avalon-S/PInVerify-Qwen3VL-4B` |

Every adapter reported in the paper goes into **one** model repository, under
subdirectories, rather than one repository per variant.

---

## Before anything

Uploads need direct access to huggingface.co; read-only mirrors cannot receive
data. `prepare_hf_release.py` probes the Hub first and tells you when it cannot
reach it.

```bash
huggingface-cli login            # or export HF_TOKEN=...
```

If your machine reaches the Hub through a proxy, prefix each command below with
it: a non-interactive SSH session does not read `.bashrc`, so it will not already
be in the environment. Proxied transfers drop often enough to matter, and
`huggingface_hub` resumes an interrupted one on the next run, so run long uploads
under `tmux` or `nohup`.

---

## Dataset

The test split and the training pools live on **different machines**, so the
dataset ships in two stages with no copying between them.

### Stage 1: test split

On the evaluation machine, where `pin_capture/val` is complete:

```bash
DATA=<dataset root on this machine>

python scripts/prepare_hf_release.py check \
    --data-root "$DATA" --splits val

python scripts/prepare_hf_release.py upload \
    --data-root "$DATA" --splits val \
    --repo Avalon-S/PInVerify --card hf_cards/dataset_card.md
```

`--splits val` skips `train_sft/`, `train_rl/` and `pin_capture/train*/`, so a
machine that lacks them is not a problem.

### Stage 2: training pools

On the training machine, where `train_sft` and `train_rl` are complete:

```bash
DATA=<dataset root on this machine>

python scripts/prepare_hf_release.py check \
    --data-root "$DATA" --splits train

python scripts/prepare_hf_release.py upload \
    --data-root "$DATA" --splits train \
    --repo Avalon-S/PInVerify
```

Leave `--card` off the second push so it does not overwrite the card uploaded in
stage 1.

`check` walks every index row and refuses to publish when episode directories
are missing, which is the failure mode that matters here.

---

## Models

All adapters go to `Avalon-S/PInVerify-Qwen3VL-4B` under subdirectories. From
the training machine:

```bash
REPO=Avalon-S/PInVerify-Qwen3VL-4B
T=<directory holding the training outputs>

# Card first, so the repo has a front page
python scripts/prepare_hf_release.py upload-model \
    --adapter $T/gspo_v2_sft_output/v0-20260304-020831/checkpoint-500 \
    --repo $REPO --card hf_cards/model_card.md \
    --path-in-repo generic-cot/gspo

# The rest
for spec in \
  "generic-cot/sft:$T/sft_v2_output/v0-20260303-204441/checkpoint-686" \
  "generic-cot/grpo:$T/grpo_v2_sft_output/v0-20260303-230946/checkpoint-500" \
  "specific-cot/sft:$T/sft_v3_output/v0-20260304-050642/checkpoint-686" \
  "specific-cot/dpo-200:$T/dpo_v3_output/v0-20260304-075505/checkpoint-200" \
  "specific-cot/dpo-400:$T/dpo_v3_output/v0-20260304-075505/checkpoint-400" \
  "specific-cot/grpo:$T/grpo_v3_output/v1-20260304-092102/checkpoint-500" \
  "specific-cot/gspo:$T/gspo_v3_output/v0-20260304-134332/checkpoint-500" ; do
    dest="${spec%%:*}"; src="${spec#*:}"
    python scripts/prepare_hf_release.py upload-model \
        --adapter "$src" --repo "$REPO" --path-in-repo "$dest"
done
```

Each adapter is about 127 MB, so all eight together are around 1 GB.

`upload-model` prints the base model, LoRA rank and alpha from
`adapter_config.json` before pushing, so a wrong path is obvious immediately. It
also carries `args.json` along, which is what makes each run reproducible.

The `images/` directory inside a training output holds ms-swift's loss curves and
is skipped.

---

## After upload

1. Check each repo renders: YAML frontmatter, tags, license, `base_model`.
2. Link the paper: the arXiv ID in the card should surface the paper widget. If
   not, add it through "Edit metadata" on the repo page.
3. Confirm the dataset viewer works on `val/`.
4. Submit at [hf.co/papers/submit](https://huggingface.co/papers/submit), claim
   the paper, and link both repos from the paper page.

## Why one model repo

Separate repositories give per-variant download counts, which is the usual
advice. This project takes the other trade: eight adapters differing only in
training stage are easier to compare side by side in one place, and the paper
treats them as one family. The subdirectory layout keeps them individually
downloadable:

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B \
    --include "generic-cot/gspo/*" --local-dir ./models/pinverify
```
