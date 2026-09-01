#!/usr/bin/env python3
"""
Verify a local pv_dataset tree and push it to the Hugging Face Hub.

The dataset is assembled on the GPU machine that ran the captures, so the usual
failure mode when releasing is uploading a partial copy: a few scenes synced to
a laptop, crops never copied back. `check` walks every split index and reports
exactly what is missing before anything is pushed.

Usage:
    # 1. Verify the tree is complete
    python scripts/prepare_hf_release.py check --data-root ./data/pv_dataset

    # 2. Push it (runs check first and refuses on missing files)
    huggingface-cli login
    python scripts/prepare_hf_release.py upload \
        --data-root ./data/pv_dataset \
        --repo Avalon-S/PInVerify \
        --card hf_cards/dataset_card.md

    # 3. Or push a LoRA adapter
    python scripts/prepare_hf_release.py upload-model \
        --adapter ./outputs/training/gspo_v2_from_sft/checkpoint-500 \
        --repo Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO \
        --card hf_cards/model_card_sft_gspo.md

Both upload paths probe the Hub first and say so when it is unreachable. If your
machine reaches huggingface.co through a proxy, put it in the same command as the
upload: a non-interactive SSH session does not read .bashrc.

Expected layout under --data-root:

    pin_capture/val/<scene>/<episode>/{meta.json,rgb/,mask/,overview.png}
    pin_capture/train_sft/<scene>/<episode>/...
    pin_capture/train_rl/<scene>/<episode>/...
    image_gt/<category>/*.png
    val/pv_index_{50,100,500,1000,all}.jsonl
    train_sft/{pv_train_sft_index.jsonl,sft_data_v2.jsonl,sft_data_v3.jsonl,crops/,crops_v3/}
    train_rl/{pv_train_rl_index.jsonl,rl_data_v2.jsonl,dpo_data_v3.jsonl,crops_rl/,crops_dpo/}
    {attr,category,merge}_cache.json
    object_descriptions_with_category.json
"""

import argparse
import json
import os
import sys

# split -> (index file, human label)
INDEX_SPECS = {
    "val":       ("val/pv_index_all.jsonl", "val (3,000-episode test split)"),
    "train_sft": ("train_sft/pv_train_sft_index.jsonl", "train_sft pool"),
    "train_rl":  ("train_rl/pv_train_rl_index.jsonl", "train_rl pool"),
}

# The release ships in stages: the test split first, the training pools later.
STAGES = {
    "val": ["val"],
    "train": ["train_sft", "train_rl"],
    "all": ["val", "train_sft", "train_rl"],
}

REQUIRED_FILES = [
    "attr_cache.json",
    "category_cache.json",
    "merge_cache.json",
    "object_descriptions_with_category.json",
    "val/pv_index_50.jsonl",
    "val/pv_index_all.jsonl",
]

REQUIRED_DIRS = [
    "image_gt",
    "pin_capture",
]

# Large directories that are easy to forget when copying between machines.
BULK_DIRS = {
    "val": [],
    "train_sft": ["train_sft/crops", "train_sft/crops_v3"],
    "train_rl": ["train_rl/crops_rl", "train_rl/crops_dpo"],
}


def check_episodes(root, index_rel, label, sample_limit=None):
    """Walk one index and report missing episode dirs / empty rgb dirs."""
    path = os.path.join(root, index_rel)
    if not os.path.isfile(path):
        return {"label": label, "index": index_rel, "status": "index missing",
                "total": 0, "missing_dir": 0, "empty_rgb": 0, "examples": []}

    total = missing_dir = empty_rgb = 0
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ep = row.get("episode_path")
            if not ep:
                continue
            if sample_limit and total >= sample_limit:
                break
            total += 1
            abs_ep = os.path.join(root, ep)
            if not os.path.isdir(abs_ep):
                missing_dir += 1
                if len(examples) < 3:
                    examples.append(ep)
                continue
            rgb_dir = os.path.join(abs_ep, "rgb")
            if not os.path.isdir(rgb_dir) or not os.listdir(rgb_dir):
                empty_rgb += 1
                if len(examples) < 3:
                    examples.append(ep + " (no rgb)")

    ok = missing_dir == 0 and empty_rgb == 0
    return {"label": label, "index": index_rel, "status": "ok" if ok else "INCOMPLETE",
            "total": total, "missing_dir": missing_dir, "empty_rgb": empty_rgb,
            "examples": examples}


def hub_reachable(timeout=12):
    """Probe the Hub. Returns (ok, detail)."""
    try:
        import urllib.request
        req = urllib.request.Request('https://huggingface.co/api/models?limit=1',
                                     headers={'User-Agent': 'pinverify-release'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, 'HTTP %d' % r.status
    except Exception as e:
        return False, '%s: %s' % (type(e).__name__, e)


def require_hub(args):
    """Fail early and usefully when the Hub cannot be reached."""
    ok, detail = hub_reachable()
    print('Hub reachable: %s (%s)' % ('yes' if ok else 'NO', detail))
    if ok:
        return True
    print()
    print('huggingface.co is not reachable from this machine.')
    print('Uploads need direct access; the read-only mirrors cannot receive data.')
    print()
    print('Set https_proxy to something that can reach huggingface.co, in this')
    print('same shell: a non-interactive SSH session does not read .bashrc, so')
    print('it has to be part of the same command as the upload.')
    return False


def count_files(path):
    n = 0
    for _, _, files in os.walk(path):
        n += len(files)
    return n


def cmd_check(args):
    root = args.data_root
    print("Checking dataset at: %s\n" % os.path.abspath(root))

    problems = 0

    print("Required files")
    for rel in REQUIRED_FILES:
        ok = os.path.isfile(os.path.join(root, rel))
        problems += 0 if ok else 1
        print("  [%s] %s" % ("ok     " if ok else "MISSING", rel))

    print("\nRequired directories")
    for rel in REQUIRED_DIRS:
        p = os.path.join(root, rel)
        ok = os.path.isdir(p)
        problems += 0 if ok else 1
        suffix = "  (%d files)" % count_files(p) if ok else ""
        print("  [%s] %s%s" % ("ok     " if ok else "MISSING", rel, suffix))

    splits = STAGES[args.splits]
    bulk = [d for s in splits for d in BULK_DIRS[s]]
    if bulk:
        print("\nBulk directories (training crops)")
        for rel in bulk:
            p = os.path.join(root, rel)
            if not os.path.isdir(p):
                problems += 1
                print("  [MISSING] %s" % rel)
                continue
            n = count_files(p)
            if n == 0:
                problems += 1
                print("  [EMPTY  ] %s  (directory exists but holds no files)" % rel)
            else:
                print("  [ok     ] %s  (%d files)" % (rel, n))

    print("\nEpisode coverage per index  (splits: %s)" % ", ".join(splits))
    for split in splits:
        index_rel, label = INDEX_SPECS[split]
        r = check_episodes(root, index_rel, label, args.sample)
        tag = "ok  " if r["status"] == "ok" else "FAIL"
        print("  [%s] %s" % (tag, label))
        print("         index: %s   rows checked: %d" % (r["index"], r["total"]))
        if r["status"] == "index missing":
            problems += 1
            print("         index file not found")
            continue
        if r["missing_dir"] or r["empty_rgb"]:
            problems += 1
            print("         missing episode dirs: %d   episodes without rgb: %d"
                  % (r["missing_dir"], r["empty_rgb"]))
            for e in r["examples"]:
                print("           e.g. %s" % e)

    print()
    if problems:
        print("FAILED: %d problem(s). This tree is not complete enough to publish." % problems)
        print("Sync the missing pieces from the machine that generated the captures.")
        return 1
    print("OK: dataset tree looks complete.")
    return 0


def cmd_upload(args):
    if not args.allow_incomplete:
        rc = cmd_check(args)
        print()
        if rc != 0:
            print("Refusing to upload an incomplete dataset.")
            print("Fix the problems above, or pass --allow-incomplete if this is deliberate.")
            return rc

    if not require_hub(args):
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed:  pip install huggingface_hub")
        return 1

    api = HfApi()
    repo = args.repo
    print("Creating/looking up dataset repo: %s" % repo)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=args.private)

    if args.card:
        if not os.path.isfile(args.card):
            print("Card not found: %s" % args.card)
            return 1
        print("Uploading card %s as README.md" % args.card)
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")

    ignore = ["*.pyc", "__pycache__/*", ".ipynb_checkpoints/*"]
    if args.splits == "val":
        # Stage 1: test split only. Skip the training side even if partially present.
        ignore += ["pin_capture/train*/**", "train_sft/**", "train_rl/**"]
    print("Uploading %s (this takes a while)" % os.path.abspath(args.data_root))
    print("  splits: %s" % ", ".join(STAGES[args.splits]))
    api.upload_folder(folder_path=args.data_root, repo_id=repo, repo_type="dataset",
                      ignore_patterns=ignore,
                      commit_message=args.message)
    print("\nDone: https://huggingface.co/datasets/%s" % repo)
    return 0


ADAPTER_FILES = ['adapter_model.safetensors', 'adapter_config.json']


def cmd_upload_model(args):
    """Push one LoRA adapter directory to its own model repo."""
    d = args.adapter
    if not os.path.isdir(d):
        print('Adapter directory not found: %s' % d)
        return 1

    missing = [f for f in ADAPTER_FILES if not os.path.isfile(os.path.join(d, f))]
    if missing:
        print('Not a LoRA adapter directory, missing: %s' % ', '.join(missing))
        print('Point --adapter at the checkpoint itself, e.g. .../checkpoint-500')
        return 1

    # the sibling metadata is what makes the release reproducible
    extras = [f for f in ('args.json', 'additional_config.json', 'trainer_state.json',
                          'training_args.bin', 'README.md')
              if os.path.isfile(os.path.join(d, f))]
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
               if os.path.isfile(os.path.join(d, f)))
    print('Adapter: %s' % os.path.abspath(d))
    print('  files: %s' % ', '.join(sorted(ADAPTER_FILES + extras)))
    print('  size:  %.1f MB' % (size / 1e6))

    cfg_path = os.path.join(d, 'adapter_config.json')
    try:
        with open(cfg_path, encoding='utf-8') as fh:
            cfg = json.load(fh)
        print('  base:  %s' % cfg.get('base_model_name_or_path'))
        print('  lora:  r=%s alpha=%s' % (cfg.get('r'), cfg.get('lora_alpha')))
    except Exception as e:
        print('  (could not read adapter_config.json: %s)' % e)

    if not require_hub(args):
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print('huggingface_hub is not installed:  pip install huggingface_hub')
        return 1

    api = HfApi()
    print('\nCreating/looking up model repo: %s' % args.repo)
    api.create_repo(repo_id=args.repo, repo_type='model', exist_ok=True, private=args.private)

    if args.card:
        if not os.path.isfile(args.card):
            print('Card not found: %s' % args.card)
            return 1
        print('Uploading card %s as README.md' % args.card)
        api.upload_file(path_or_fileobj=args.card, path_in_repo='README.md',
                        repo_id=args.repo, repo_type='model')

    print('Uploading adapter')
    api.upload_folder(folder_path=d, repo_id=args.repo, repo_type='model',
                      ignore_patterns=['*.pyc', '__pycache__/*', 'images/*'],
                      commit_message=args.message)
    print('\nDone: https://huggingface.co/%s' % args.repo)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="./data/pv_dataset",
                        help="Root of the pv_dataset tree (default: ./data/pv_dataset)")
    common.add_argument("--sample", type=int, default=None,
                        help="Only check the first N rows of each index (quick pass)")
    common.add_argument("--splits", choices=sorted(STAGES), default="all",
                        help="Which splits to verify/publish: val (stage 1), "
                             "train (the two training pools), or all (default)")

    p_check = sub.add_parser("check", parents=[common], help="Verify the local tree")
    p_check.set_defaults(func=cmd_check)

    p_up = sub.add_parser("upload", parents=[common], help="Verify, then push to the Hub")
    p_up.add_argument("--repo", required=True, help="Target dataset repo, e.g. Avalon-S/PInVerify")
    p_up.add_argument("--card", default="hf_cards/dataset_card.md",
                      help="Markdown card uploaded as README.md")
    p_up.add_argument("--private", action="store_true", help="Create the repo private")
    p_up.add_argument("--message", default="Upload PInVerify dataset",
                      help="Commit message")
    p_up.add_argument("--allow-incomplete", action="store_true",
                      help="Skip the completeness gate (not recommended)")
    p_up.set_defaults(func=cmd_upload)

    p_model = sub.add_parser("upload-model", help="Push one LoRA adapter to a model repo")
    p_model.add_argument("--adapter", required=True,
                         help="Adapter checkpoint directory, e.g. .../checkpoint-500")
    p_model.add_argument("--repo", required=True,
                         help="Target model repo, e.g. Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO")
    p_model.add_argument("--card", default=None, help="Markdown card uploaded as README.md")
    p_model.add_argument("--private", action="store_true", help="Create the repo private")
    p_model.add_argument("--path-in-repo", default=None, dest="path_in_repo",
                         help="Subdirectory inside the repo, so several adapters "
                              "can share one repository (e.g. generic-cot/gspo)")
    p_model.add_argument("--message", default="Upload LoRA adapter", help="Commit message")
    p_model.set_defaults(func=cmd_upload_model)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
