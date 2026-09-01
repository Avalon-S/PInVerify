#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
runner.py
---------------------------------
PInVerify batch evaluation driver: index-driven, with a pluggable method class.

Semantics:
1. The imagery comes from the episode of target_object_id (episode_path / scene / episode).
   Its meta/rgb/depth are loaded; the instance actually captured there is target_object_id.
2. The description comes from query_object_id.
   The prompt is built from its description and asks whether that is what the image shows.
3. label:
   1 -> same instance (positive)
   0 -> different instance (neg_same / neg_diff)
4. object_id in the results is target_object_id, i.e. the instance actually being looked at.
5. Output layout (no target_object_id level):
   <outdir>/<pair_type>/<scene>/<episode>/episode.json
6. When writing episode.json:
   ep_json["descriptions"] is dropped to avoid duplication
   meta_info is injected:
        target_object_id / category / descriptions
        query_object_id  / category / descriptions
   Both target_descriptions and query_descriptions are stored so they can be compared.
7. batch_summary.json still aggregates accuracy and failure cases.
"""

import os, argparse, random, time, importlib.util, traceback
from typing import Any, Dict, List

from tqdm import tqdm
from multiprocessing import Process, Queue

# local modules
import dataset as D
import results as R


# ===== Dynamically load any method class =====
def dynamic_import_class(file_path: str, class_name: str):
    """Load a class from a given file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Method file not found: {file_path}")
    mod_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")
    return getattr(module, class_name)


# ===== Arguments =====
def parse_args():
    ap = argparse.ArgumentParser(description="PInVerify Runner (final semantics + annotated episode.json)")

    # --- data ---
    ap.add_argument("--dataset-root", type=str, default=D.DEFAULT_DATASET_ROOT,
                    help=f"dataset root (default {D.DEFAULT_DATASET_ROOT})")
    ap.add_argument("--capture-subdir", type=str, default=D.DEFAULT_CAPTURE_SUBDIR,
                    help=f"capture subdirectory (default {D.DEFAULT_CAPTURE_SUBDIR})")
    ap.add_argument("--split", type=str, default=D.DEFAULT_SPLIT,
                    help=f"split name (default {D.DEFAULT_SPLIT})")
    ap.add_argument("--index", type=str, default=D.DEFAULT_INDEX,
                    help=f"index file path (json/jsonl/json.gz)")
    ap.add_argument("--desc-db", type=str, default=D.DEFAULT_DESC_DB,
                    help=f"path to the description database JSON")

    # --- behaviour ---
    ap.add_argument("--mode", type=str, choices=["all", "random"], default="all",
                    help="sampling mode: all | random")
    ap.add_argument("--num", type=int, default=200,
                    help="number of samples when --mode=random")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="cap on the number of pairs (0 means no cap)")

    # --- output ---
    ap.add_argument("--outdir", type=str, default="./pv_out",
                    help="output root directory")
    ap.add_argument("--save_viz", action="store_true",
                    help="save visualizations and per-step intermediates")

    # --- method selection (pluggable) ---
    ap.add_argument("--method-file", type=str, default="./methods_qwen_vl.py",
                    help="path to the .py file holding the method class")
    ap.add_argument("--method-class", type=str, default="QwenVLMethod",
                    help="name of the method class inside that file")

    # --- Qwen server endpoints (used by the method class) ---
    ap.add_argument("--qwen-text-url", type=str, default="http://127.0.0.1:12182/qwen-text",
                    help="Qwen text endpoint URL")
    ap.add_argument("--qwen-vl-url", type=str, default="http://127.0.0.1:12182/qwen-vl",
                    help="Qwen vision-language endpoint URL")

    # --- policy, cropping and inference details ---
    ap.add_argument("--use-category", action="store_true",
                    help="pass query_object_category to the model as class_text, a coarse category hint")
    ap.add_argument("--max-steps", type=int, default=3,
                    help="steps per episode (used by the multi-view methods)")
    ap.add_argument("--crop-mode", type=str, choices=["tight", "expand"], default="tight",
                    help="verification crop mode: tight | expand")
    ap.add_argument("--pad", type=int, default=3,
                    help="crop padding in pixels for tight mode")
    ap.add_argument("--min-side", type=int, default=320,
                    help="minimum short side in pixels after an expand crop")
    ap.add_argument("--attr-k", type=int, default=8,
                    help="max attributes per description in the attribute methods (default 8)")

    # --- detector mode ---
    ap.add_argument("--detector-mode",
                    type=str,
                    choices=["gdino", "bbox"],
                    default="gdino",
                    help="box source: gdino = GroundingDINO detection, bbox = mask_bbox_xyxy from meta.json")

    # --- coarse category cache (used by some methods) ---
    ap.add_argument("--coarse-cache", type=str, default="",
                    help="path to the coarse category cache JSON (optional)")

    # --- scoring ---
    ap.add_argument("--unsure-as-negative", action="store_true",
                    help="count Unsure as negative (0); otherwise Unsure episodes are excluded from the metrics")

    # --- parallelism ---
    ap.add_argument("--num-workers", type=int, default=1,
                    help="worker processes (1 means single process)")
    ap.add_argument("--devices", type=str, default="",
                    help="comma-separated GPU ids, e.g. '0,1,2,3'; empty leaves CUDA_VISIBLE_DEVICES alone")
    return ap.parse_args()


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# ===== Episode path resolution =====
def _episode_rel_from_index_record(rec: Dict[str, Any]) -> str:
    """
    Normalize an index record's episode into the short "val/<scene>/<episode>" form,
    used for the output layout and log lines.
    """
    ep_path = rec.get("episode_path")
    if isinstance(ep_path, str) and ep_path:
        parts = ep_path.strip("/").split("/")
        # typical: ["pin_capture", "val", "<scene>", "<episode>"]
        if len(parts) >= 4:
            return "/".join(parts[1:])  # -> "val/<scene>/<episode>"
        return "/".join(parts)

    # fallback, should not normally be reached
    split  = str(rec.get("split")  or "val").strip("/")
    scene  = str(rec.get("scene")  or rec.get("scene_id") or "").strip("/")
    ep_id  = rec.get("episode")    or rec.get("episode_id") or rec.get("ep_id") or ""
    ep_id  = str(ep_id).strip("/")
    return f"{split}/{scene}/{ep_id}"


def _episode_abs_dir(dataset_root: str, rec: Dict[str, Any]) -> str:
    """
    Resolve the absolute on-disk path of the episode for this record.
    episode_path takes precedence.
    """
    ep_rel = rec.get("episode_path")
    if isinstance(ep_rel, str) and ep_rel:
        return os.path.join(dataset_root, ep_rel)

    # fallback
    split  = str(rec.get("split")  or "val").strip("/")
    scene  = str(rec.get("scene")  or rec.get("scene_id") or "").strip("/")
    ep_id  = rec.get("episode")    or rec.get("episode_id") or rec.get("ep_id") or ""
    ep_id  = str(ep_id).strip("/")
    return os.path.join(dataset_root, "pin_capture", split, scene, ep_id)


# ===== Bucket the results into correct / wrong
def _group_results_by_type_and_correctness(res_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in res_list:
        pt = r.get("pair_type", "unknown")
        is_correct = (int(r["pred"]) == int(r["label"]))
        bucket = "correct" if is_correct else "wrong"

        if pt not in grouped:
            grouped[pt] = {"correct": [], "wrong": []}
        grouped[pt][bucket].append(r)
    return grouped


def _save_episode_json(out_ep: str,
                       ep_json: Dict[str, Any],
                       target_object_id: str,
                       target_object_cat: str,
                       target_descs: List[str],
                       query_object_id: str,
                       query_object_cat: str,
                       query_descs: List[str]):
    """
    Write episode.json:
    - drop ep_json["descriptions"], which duplicates meta_info
    - drop ep_json["step"]["class_gate"]: coarse-category gating is gone, so the field is a leftover
    - inject meta_info with the target/query id, category and descriptions
    """

    # copy so the caller's object is not mutated
    safe_json = dict(ep_json)

    # 1. skip the descriptions field; both sides go into meta_info instead
    if "descriptions" in safe_json:
        del safe_json["descriptions"]

    # 2. drop step.class_gate if present
    step_block = safe_json.get("step")
    if isinstance(step_block, dict) and "class_gate" in step_block:
        del step_block["class_gate"]

    # 3. inject the comparison metadata
    safe_json["meta_info"] = {
        "target_object_id": target_object_id,
        "target_object_category": target_object_cat,
        "target_descriptions": target_descs,

        "query_object_id": query_object_id,
        "query_object_category": query_object_cat,
        "query_descriptions": query_descs
    }

    # 4. write to disk
    ensure_dir(out_ep)
    with open(os.path.join(out_ep, "episode.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(safe_json, f, ensure_ascii=False, indent=2)



def _run_one_pair(rec: Dict[str, Any],
                  args,
                  method,
                  desc_db: Dict[str, Any],
                  rng_for_worker_random: random.Random,
                  idx_for_name: str = "") -> Dict[str, Any]:
    """
    Core evaluation of a single pair.

    Semantics:
    - target_object_id / target_object_category:
        the object actually captured in this episode, i.e. the instance in the image.
    - query_object_id / query_object_category:
        the object being searched for, whose description drives the prompt.
    - The model decides whether the target_object_id in the image is the query_object_id.
    - label:
        1 -> same instance (positive)
        0 -> different instance (neg_same / neg_diff)

    The return value feeds the summary and batch_summary.json.
    """

    pair_type   = rec.get("pair_type", "unknown")
    label_raw   = rec.get("label", None)
    if label_raw is None:
        raise RuntimeError("Record missing label")
    label_int   = int(label_raw)

    # 1. locate the episode directory (imagery comes from target_object_id)
    episode_abs = _episode_abs_dir(args.dataset_root, rec)
    if not os.path.isdir(episode_abs):
        raise FileNotFoundError(f"episode_abs missing: {episode_abs}")

    # 2. read meta.json and its captures
    meta = D.load_episode_from_root(episode_abs)

    # 3. which instance is in the image
    target_object_id  = str(rec.get("target_object_id") or "")
    target_object_cat = str(rec.get("target_object_category") or "")

    # 4. which instance is being searched for (query / prompt)
    query_object_id   = str(rec.get("query_object_id") or "")
    query_object_cat  = str(rec.get("query_object_category") or "")

    # 5. fetch both descriptions from the database
    #    target_descriptions: description of the instance in the image
    #    query_descriptions : description of the searched instance, the one shown to the model
    target_descs = D.get_descs_for_object(desc_db, target_object_id, pad_to=3)
    query_descs  = D.get_descs_for_object(desc_db, query_object_id,  pad_to=3)

    # 6. build class_text: with --use-category the model is told which coarse category to expect
    if args.use_category:
        class_text = query_object_cat or desc_db.get(query_object_id, {}).get("object_category", "") or ""
    else:
        class_text = ""

    # 7. output directory (no target_object_id level any more)
    episode_rel = _episode_rel_from_index_record(rec)
    parts = episode_rel.strip("/").split("/")
    if len(parts) >= 3:
        scene_name, ep_id = parts[-2], parts[-1]
        out_ep = os.path.join(
            args.outdir,
            pair_type,
            scene_name,
            ep_id,
        )
    else:
        out_ep = os.path.join(
            args.outdir,
            pair_type,
            f"{idx_for_name or 'unk'}",
        )
    ensure_dir(out_ep)

    # 8. reproducible random value for methods that pick near/far at random
    rng_val = rng_for_worker_random.randint(0, 2**31 - 1)
    _ = rng_val  # kept as a placeholder for methods that need it

    # 9. run the method's multi-step inference
    #    query_descs is passed as raw_descs because the question is
    #    whether the image shows query_object_id
    ep_json = method.run_episode(
        meta=meta,
        class_text=class_text,
        raw_descs=query_descs,
        outdir=out_ep,
        args=args
    )

    # 10. write episode.json with the target/query metadata and both descriptions
    _save_episode_json(
        out_ep=out_ep,
        ep_json=ep_json,
        target_object_id=target_object_id,
        target_object_cat=target_object_cat,
        target_descs=target_descs,
        query_object_id=query_object_id,
        query_object_cat=query_object_cat,
        query_descs=query_descs
    )

    # 11. parse the final decision
    final_block = ep_json.get("final") or {}
    decision = (final_block.get("decision") or "").strip().title()  # "Yes"/"No"/"Unsure"

    if decision == "Yes":
        pred = 1
    elif decision == "No":
        pred = 0
    else:
        # Unsure: unless --unsure-as-negative is set, drop the episode from the summary
        pred = 0 if args.unsure_as_negative else None

    if pred is None:
        # excluded from the summary
        raise RuntimeError("Prediction is None (Unsure not counted)")

    # 12. hand the result to the summary
    return {
        "pred": int(pred),
        "label": label_int,
        "pair_type": pair_type,
        "episode_rel": episode_rel,
        "scene": parts[-2] if len(parts) >= 2 else "",
        "episode_id": parts[-1] if len(parts) >= 1 else "",
        # the instance actually being looked at
        "object_id": target_object_id,
    }


def _run_shard(shard_id: int,
               pairs_slice: List[Dict[str, Any]],
               args,
               device: str,
               desc_db: Dict[str, Any],
               out_queue: Queue):
    """
    One worker process evaluates one shard of pairs:
      - after each pair: out_queue.put({"type": "tick", "shard": id})
      - when finished: out_queue.put({"type": "done", "shard": id, "results": [...]}).
    """

    # pin the device (CUDA_VISIBLE_DEVICES)
    if device != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    print(f"[Worker {shard_id}] start  pid={os.getpid()}  pairs={len(pairs_slice)}  device='{device}'")

    rng_for_worker = random.Random(args.seed + shard_id * 9973)

    # load the method class
    MethodClass = dynamic_import_class(args.method_file, args.method_class)
    try:
        method = MethodClass(args.qwen_text_url, args.qwen_vl_url)
    except TypeError:
        # some implementations do not need both URLs
        method = MethodClass()

    partial_results: List[Dict[str, Any]] = []

    for loc_idx, rec in enumerate(pairs_slice):
        try:
            row = _run_one_pair(
                rec=rec,
                args=args,
                method=method,
                desc_db=desc_db,
                rng_for_worker_random=rng_for_worker,
                idx_for_name=f"sh{shard_id}_{loc_idx:05d}"
            )
            partial_results.append(row)
        except Exception as e:
            print(f"[Worker {shard_id}][WARN] Failed pair idx={loc_idx}: {e}")
            print(traceback.format_exc())

        out_queue.put({"type": "tick", "shard": shard_id})

    print(f"[Worker {shard_id}] done   pid={os.getpid()}  results={len(partial_results)}")
    out_queue.put({"type": "done", "shard": shard_id, "results": partial_results})


def main():
    args = parse_args()
    random.seed(args.seed)

    # resolve the index and desc_db paths
    index_path = args.index if os.path.isabs(args.index) else os.path.join(args.dataset_root, args.split, args.index)
    desc_path  = args.desc_db if os.path.isabs(args.desc_db) else os.path.join(args.dataset_root, args.split, args.desc_db)

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"Desc DB not found: {desc_path}")

    ensure_dir(args.outdir)

    # load pairs and the description database
    pairs = D.load_pairs(index_path, mode=args.mode, num=args.num, seed=args.seed)
    desc_db = D.load_desc_db(desc_path)

    if args.max_episodes and len(pairs) > args.max_episodes:
        pairs = pairs[:args.max_episodes]

    print(f"[Runner] Loaded {len(pairs)} pairs from {index_path}")
    print(f"[Runner] Desc DB: {desc_path}")
    print(f"[Runner] Method: {args.method_file}::{args.method_class}")
    print(f"[Runner] detector_mode = {args.detector_mode}")
    print(f"[Runner] crop_mode     = {args.crop_mode}, pad={args.pad}, min_side={args.min_side}")
    if args.coarse_cache:
        print(f"[Runner] Coarse cache (arg): {args.coarse_cache}")
    else:
        print(f"[Runner] Coarse cache: (method internal default or env)")
    if args.save_viz:
        print(f"[Runner] save_viz is ON")

    # ===== Single-process path =====
    if args.num_workers <= 1:
        print("[Runner] Mode: single-process")

        MethodClass = dynamic_import_class(args.method_file, args.method_class)
        try:
            method = MethodClass(args.qwen_text_url, args.qwen_vl_url)
        except TypeError:
            method = MethodClass()

        cls_results: List[Dict[str, Any]] = []
        t_start = time.time()

        rng_for_worker = random.Random(args.seed)

        for idx, rec in enumerate(tqdm(pairs, desc="[Runner] Evaluating", dynamic_ncols=True)):
            try:
                row = _run_one_pair(
                    rec=rec,
                    args=args,
                    method=method,
                    desc_db=desc_db,
                    rng_for_worker_random=rng_for_worker,
                    idx_for_name=f"{idx:05d}"
                )
                cls_results.append(row)
            except Exception as e:
                print(f"[Runner][WARN] Failed on pair #{idx}: {e}")
                print(traceback.format_exc())
                continue

        if cls_results:
            summary = R.summarize_classification(cls_results)
            R.print_cls_summary(summary)

            results_bucketed = _group_results_by_type_and_correctness(cls_results)

            R.save_json({
                "args": vars(args),
                "summary": summary,
                "results": results_bucketed
            }, os.path.join(args.outdir, "batch_summary.json"))
        else:
            print("[Runner] No valid classification results to summarize "
                  "(maybe all were Unsure and --unsure-as-negative is OFF).")

        print(f"[Runner] Done in {time.time() - t_start:.2f}s. Output → {args.outdir}")
        return

    # ===== Multi-process path =====
    num_workers = max(2, int(args.num_workers))

    # build the per-worker GPU binding list
    if args.devices.strip():
        devices = [d.strip() for d in args.devices.split(",") if d.strip() != ""]
        if len(devices) < num_workers:
            print(f"[Runner][WARN] devices={len(devices)} < num_workers={num_workers}, will round-robin reuse")
        print(f"[Runner] Mode: multi-process GPUs={devices} workers={num_workers}")
    else:
        devices = [""] * num_workers
        print(f"[Runner] Mode: multi-process (single GPU/CPU) workers={num_workers} device=inherit")

    # split pairs evenly into shards
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(pairs):
        shards[i % num_workers].append(item)

    counts = [len(s) for s in shards]
    print("[Runner] Shards: " + ", ".join([f"#{i}:{c}" for i, c in enumerate(counts)]))

    q = Queue()
    procs: List[Process] = []
    t0 = time.time()

    # start the workers
    for sid in range(num_workers):
        dev = devices[sid % len(devices)]
        p = Process(
            target=_run_shard,
            args=(sid, shards[sid], args, dev, desc_db, q)
        )
        p.start()
        procs.append(p)

    total_pairs = len(pairs)
    pbar = tqdm(total=total_pairs, desc="[Runner] Evaluating", dynamic_ncols=True)

    cls_results_all: List[Dict[str, Any]] = []
    done_workers = 0

    # the parent merges results and shows overall progress
    while done_workers < num_workers:
        msg = q.get()
        mtype = msg.get("type", "")
        if mtype == "tick":
            pbar.update(1)
        elif mtype == "done":
            cls_results_all.extend(msg.get("results", []))
            done_workers += 1
    pbar.close()

    # wait for the workers to exit
    for p in procs:
        p.join()

    # aggregate
    if cls_results_all:
        summary = R.summarize_classification(cls_results_all)
        R.print_cls_summary(summary)

        results_bucketed = _group_results_by_type_and_correctness(cls_results_all)

        R.save_json({
            "args": vars(args),
            "summary": summary,
            "results": results_bucketed
        }, os.path.join(args.outdir, "batch_summary.json"))
    else:
        print("[Runner] No valid classification results to summarize "
              "(maybe all were Unsure and --unsure-as-negative is OFF).")

    print(f"[Runner] Done in {time.time() - t0:.2f}s. Output → {args.outdir}")


if __name__ == "__main__":
    main()
