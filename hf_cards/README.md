# Hugging Face Hub Cards

Pre-authored YAML+Markdown cards for the planned PInVerify HF Hub releases.

## Inventory

| File | Target repo |
|---|---|
| `dataset_card.md`              | dataset: `Avalon-S/PInVerify` |
| `model_card_sft.md`            | model:   `Avalon-S/PInVerify-Qwen3VL-4B-SFT` |
| `model_card_sft_dpo_200.md`    | model:   `Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-200` |
| `model_card_sft_dpo_400.md`    | model:   `Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-400` |
| `model_card_sft_grpo.md`       | model:   `Avalon-S/PInVerify-Qwen3VL-4B-SFT-GRPO` |
| `model_card_sft_gspo.md`       | model:   `Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO` ⭐ paper best |

## Upload recipe

```bash
pip install huggingface_hub
huggingface-cli login

# Dataset (1.4 GB) — push from your local autodl-tmp/pv_dataset directory
huggingface-cli upload \
  Avalon-S/PInVerify ./data/pv_dataset \
  --repo-type dataset
huggingface-cli upload \
  Avalon-S/PInVerify hf_cards/dataset_card.md README.md \
  --repo-type dataset

# Each model — repeat the pair (weights + README) for all five
huggingface-cli upload \
  Avalon-S/PInVerify-Qwen3VL-4B-SFT ./outputs/training/sft \
  --repo-type model
huggingface-cli upload \
  Avalon-S/PInVerify-Qwen3VL-4B-SFT hf_cards/model_card_sft.md README.md \
  --repo-type model

# ... repeat for SFT-DPO-200 / SFT-DPO-400 / SFT-GRPO / SFT-GSPO
```

## After upload

1. Visit each repo on the Hub and verify:
   - YAML frontmatter renders as expected (tags, license, base_model)
   - The "linked papers" widget points at [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) — if not, click "Edit metadata" and add the arXiv ID manually
   - Dataset card shows the dataset viewer for `val/`
2. Go to [hf.co/papers/submit](https://huggingface.co/papers/submit), paste the arXiv URL, claim the paper, and link all 6 repos via the paper page UI.
3. Ping @NielsRogge in the GitHub issue once the page is live so HF can amplify.

## Why separate model repos?

Per Niels Rogge (HF Open-Source team): "We encourage researchers to push each model checkpoint to a separate model repository, so that things like download stats also work."

Each repo getting independent download/like counters compounds discoverability — a single shared-repo loses per-variant signal.
