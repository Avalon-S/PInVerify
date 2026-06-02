#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
runner.py
---------------------------------
PInVerify 批量评测主程序（索引驱动 + 可插拔 method）

最终版（包含你的定制要求）：
1. 视觉来源 = target_object_id 对应的 episode（episode_path / scene / episode）。
   - 我们加载这条 episode 的 meta/rgb/depth，这一集里拍到的实体就是 target_object_id。
2. 文本描述来源 = query_object_id。
   - 我们用 query_object_id 的描述生成 prompt，问模型“画面里是它吗？”
3. label:
   - 1 → 同一实例 (positive)
   - 0 → 不是同一实例 (neg_same / neg_diff)
4. 统计结果里的 object_id = target_object_id（我们实际在看的实例是谁）。
5. 输出目录结构（不再包含 target_object_id 这一级）：
   <outdir>/<pair_type>/<scene>/<episode>/episode.json
6. 写 episode.json 时：
   - 移除 ep_json["descriptions"]（避免冗余/混淆）
   - 注入 meta_info：
        target_object_id / category / descriptions
        query_object_id  / category / descriptions
   其中 target_descriptions / query_descriptions 都会写进去，方便对照。
7. 仍然生成 batch_summary.json 汇总准确率、错误样例等。
"""

import os, argparse, random, time, importlib.util, traceback
from typing import Any, Dict, List

from tqdm import tqdm
from multiprocessing import Process, Queue

# 本地模块
import dataset as D
import results as R


# ===== 动态加载任意 Method 类 =====
def dynamic_import_class(file_path: str, class_name: str):
    """从指定文件中加载类"""
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


# ===== 参数定义 =====
def parse_args():
    ap = argparse.ArgumentParser(description="PInVerify Runner (final semantics + annotated episode.json)")

    # --- 数据相关 ---
    ap.add_argument("--dataset-root", type=str, default=D.DEFAULT_DATASET_ROOT,
                    help=f"数据根目录（默认 {D.DEFAULT_DATASET_ROOT}）")
    ap.add_argument("--capture-subdir", type=str, default=D.DEFAULT_CAPTURE_SUBDIR,
                    help=f"采集子目录名（默认 {D.DEFAULT_CAPTURE_SUBDIR}）")
    ap.add_argument("--split", type=str, default=D.DEFAULT_SPLIT,
                    help=f"split 名（默认 {D.DEFAULT_SPLIT}）")
    ap.add_argument("--index", type=str, default=D.DEFAULT_INDEX,
                    help=f"索引文件路径（json/jsonl/json.gz）")
    ap.add_argument("--desc-db", type=str, default=D.DEFAULT_DESC_DB,
                    help=f"描述库 JSON 文件路径")

    # --- 行为控制 ---
    ap.add_argument("--mode", type=str, choices=["all", "random"], default="all",
                    help="抽样模式：all | random")
    ap.add_argument("--num", type=int, default=200,
                    help="当 --mode=random 时抽样数量")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="最多评估多少个 pair（0 表示不限制）")

    # --- 输出控制 ---
    ap.add_argument("--outdir", type=str, default="./pv_out",
                    help="结果输出根目录")
    ap.add_argument("--save_viz", action="store_true",
                    help="保存可视化与中间产物（step 目录）")

    # --- method 选择（可插拔）---
    ap.add_argument("--method-file", type=str, default="./methods_qwen_vl.py",
                    help="包含方法类的 .py 文件路径")
    ap.add_argument("--method-class", type=str, default="QwenVLMethod",
                    help="方法类名（文件内的类）")

    # --- Qwen 服务接口（供方法类用）---
    ap.add_argument("--qwen-text-url", type=str, default="http://127.0.0.1:12182/qwen-text",
                    help="Qwen 文本接口 URL")
    ap.add_argument("--qwen-vl-url", type=str, default="http://127.0.0.1:12182/qwen-vl",
                    help="Qwen 图文接口 URL")

    # --- 策略与裁剪 / 推理细节 ---
    ap.add_argument("--use-category", action="store_true",
                    help="把 query_object_category 作为 class_text 传给模型（给模型一个粗类别引导）")
    ap.add_argument("--max-steps", type=int, default=3,
                    help="单 episode 运行步数（多视角方法可能会用到）")
    ap.add_argument("--crop-mode", type=str, choices=["tight", "expand"], default="tight",
                    help="核验裁剪模式：tight | expand")
    ap.add_argument("--pad", type=int, default=3,
                    help="tight 模式下裁剪 padding（像素）")
    ap.add_argument("--min-side", type=int, default=320,
                    help="expand 模式下裁剪后短边最小像素")
    ap.add_argument("--attr-k", type=int, default=8,
                    help="attr式方法里单条描述最大属性数（默认8）")

    # --- detector 模式 ---
    ap.add_argument("--detector-mode",
                    type=str,
                    choices=["gdino", "bbox"],
                    default="gdino",
                    help="目标框来源：gdino=GroundingDINO检测，bbox=直接用meta.json里的mask_bbox_xyxy")

    # --- coarse 类别缓存（方法里可能用）---
    ap.add_argument("--coarse-cache", type=str, default="",
                    help="粗类别缓存 JSON 路径（可选）")

    # --- 统计控制 ---
    ap.add_argument("--unsure-as-negative", action="store_true",
                    help="把 Unsure 当作负类(0)。否则 Unsure 样本不计入准确率/召回率计算。")

    # --- 并行控制 ---
    ap.add_argument("--num-workers", type=int, default=1,
                    help="并行进程数（1 表示单进程）")
    ap.add_argument("--devices", type=str, default="",
                    help="逗号分隔的 GPU id 列表，比如 '0,1,2,3'；留空则不绑定 CUDA_VISIBLE_DEVICES")
    return ap.parse_args()


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# ===== Episode 相对路径解析 =====
def _episode_rel_from_index_record(rec: Dict[str, Any]) -> str:
    """
    把 index 记录里的 episode 规范成 "val/<scene>/<episode>" 这种短路径，
    用于输出结构和日志显示。
    """
    ep_path = rec.get("episode_path")
    if isinstance(ep_path, str) and ep_path:
        parts = ep_path.strip("/").split("/")
        # 典型: ["pin_capture","val","<scene>","<episode>"]
        if len(parts) >= 4:
            return "/".join(parts[1:])  # -> "val/<scene>/<episode>"
        return "/".join(parts)

    # 回退方案（理论上不该走到这里）
    split  = str(rec.get("split")  or "val").strip("/")
    scene  = str(rec.get("scene")  or rec.get("scene_id") or "").strip("/")
    ep_id  = rec.get("episode")    or rec.get("episode_id") or rec.get("ep_id") or ""
    ep_id  = str(ep_id).strip("/")
    return f"{split}/{scene}/{ep_id}"


def _episode_abs_dir(dataset_root: str, rec: Dict[str, Any]) -> str:
    """
    根据 record 推出该 episode 在磁盘上的绝对路径。
    优先使用 episode_path。
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


# ===== 把评测结果分桶，统计 correct / wrong
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
    写 episode.json:
    - 去掉 ep_json["descriptions"]（避免和我们自己的 meta_info 重复）
    - 去掉 ep_json["step"]["class_gate"]（我们现在的方法不再做粗类别 gating，这个字段如果存在就视为旧逻辑的残留）
    - 注入 meta_info，记录 target/query 的 id / category / descriptions
    """

    # 复制，避免原对象被外部继续使用时被我们改坏
    safe_json = dict(ep_json)

    # 1. descriptions 字段不要写进去（我们自己会放 meta_info 里的两边描述）
    if "descriptions" in safe_json:
        del safe_json["descriptions"]

    # 2. 如果 step.class_gate 存在，删除它
    step_block = safe_json.get("step")
    if isinstance(step_block, dict) and "class_gate" in step_block:
        del step_block["class_gate"]

    # 3. 注入我们自己的对照信息
    safe_json["meta_info"] = {
        "target_object_id": target_object_id,
        "target_object_category": target_object_cat,
        "target_descriptions": target_descs,

        "query_object_id": query_object_id,
        "query_object_category": query_object_cat,
        "query_descriptions": query_descs
    }

    # 4. 写盘
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
    核心执行逻辑（单条 pair）。

    最终语义：
    - target_object_id / target_object_category:
        这条 episode 真正拍到的那个物体（画面里的实例）。
    - query_object_id / query_object_category:
        我们声称要找的物体（prompt 使用它）。
    - 模型判断：画面里的 target_object_id 是否就是 query_object_id ?
    - label:
        1 -> 同一实例 (positive)
        0 -> 不同实例 (neg_same / neg_diff)

    返回值会被用于 summary 和 batch_summary.json。
    """

    pair_type   = rec.get("pair_type", "unknown")
    label_raw   = rec.get("label", None)
    if label_raw is None:
        raise RuntimeError("Record missing label")
    label_int   = int(label_raw)

    # 1. 定位 episode 目录（视觉来源 = target_object_id 的那一集）
    episode_abs = _episode_abs_dir(args.dataset_root, rec)
    if not os.path.isdir(episode_abs):
        raise FileNotFoundError(f"episode_abs missing: {episode_abs}")

    # 2. 读 meta.json + captures
    meta = D.load_episode_from_root(episode_abs)

    # 3. 谁在画面里（视觉对象）
    target_object_id  = str(rec.get("target_object_id") or "")
    target_object_cat = str(rec.get("target_object_category") or "")

    # 4. 我们声称要找谁（query / prompt）
    query_object_id   = str(rec.get("query_object_id") or "")
    query_object_cat  = str(rec.get("query_object_category") or "")

    # 5. 从描述库取两边的描述
    #    target_descriptions: 画面里这个具体实例的描述
    #    query_descriptions : 我们要找的那只实例的描述（也是喂给模型的描述）
    target_descs = D.get_descs_for_object(desc_db, target_object_id, pad_to=3)
    query_descs  = D.get_descs_for_object(desc_db, query_object_id,  pad_to=3)

    # 6. 构建 class_text（如果开启 use-category，就告诉模型“我要找的是哪一类”）
    if args.use_category:
        class_text = query_object_cat or desc_db.get(query_object_id, {}).get("object_category", "") or ""
    else:
        class_text = ""

    # 7. 输出目录（注意：不再包含 target_object_id 这一层）
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

    # 8. 给 method 一个可复现的随机值（方法内部如果需要随机选 near/far 等）
    rng_val = rng_for_worker_random.randint(0, 2**31 - 1)
    _ = rng_val  # 我们留着占位，未来如果 method 需要我们就可以把这个塞进 args 或额外参数

    # 9. 实际调用 method 执行多步推理
    #    注意：我们把 query_descs 作为 raw_descs 传给它，
    #    因为我们的问题是“这是不是 query_object_id？”
    ep_json = method.run_episode(
        meta=meta,
        class_text=class_text,
        raw_descs=query_descs,
        outdir=out_ep,
        args=args
    )

    # 10. 把 episode.json 写到磁盘，包含 target/query 的元信息和双边描述
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

    # 11. 解析最终决策
    final_block = ep_json.get("final") or {}
    decision = (final_block.get("decision") or "").strip().title()  # "Yes"/"No"/"Unsure"

    if decision == "Yes":
        pred = 1
    elif decision == "No":
        pred = 0
    else:
        # Unsure：如果用户没让我们把Unsure当负类，就直接丢掉这条，不计入 summary
        pred = 0 if args.unsure_as_negative else None

    if pred is None:
        # 这条不进入最终 summary
        raise RuntimeError("Prediction is None (Unsure not counted)")

    # 12. 返回给 summary 使用
    return {
        "pred": int(pred),
        "label": label_int,
        "pair_type": pair_type,
        "episode_rel": episode_rel,
        "scene": parts[-2] if len(parts) >= 2 else "",
        "episode_id": parts[-1] if len(parts) >= 1 else "",
        # 我们真正在看的实例是谁
        "object_id": target_object_id,
    }


def _run_shard(shard_id: int,
               pairs_slice: List[Dict[str, Any]],
               args,
               device: str,
               desc_db: Dict[str, Any],
               out_queue: Queue):
    """
    一个进程负责跑一片 pairs：
      - 每完成 1 条 pair → out_queue.put({"type":"tick","shard":id})
      - 全部完成后 → out_queue.put({"type":"done","shard":id,"results":[...]}).
    """

    # 绑定设备 (CUDA_VISIBLE_DEVICES)
    if device != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    print(f"[Worker {shard_id}] start  pid={os.getpid()}  pairs={len(pairs_slice)}  device='{device}'")

    rng_for_worker = random.Random(args.seed + shard_id * 9973)

    # 动态加载 method
    MethodClass = dynamic_import_class(args.method_file, args.method_class)
    try:
        method = MethodClass(args.qwen_text_url, args.qwen_vl_url)
    except TypeError:
        # 有的实现也许不要求两个URL
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

    # 解析 index / desc_db 实际路径
    index_path = args.index if os.path.isabs(args.index) else os.path.join(args.dataset_root, args.split, args.index)
    desc_path  = args.desc_db if os.path.isabs(args.desc_db) else os.path.join(args.dataset_root, args.split, args.desc_db)

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"Desc DB not found: {desc_path}")

    ensure_dir(args.outdir)

    # 加载 pairs & 描述库
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

    # ===== 单进程路径 =====
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

    # ===== 多进程路径 =====
    num_workers = max(2, int(args.num_workers))

    # 准备多 GPU 绑定列表
    if args.devices.strip():
        devices = [d.strip() for d in args.devices.split(",") if d.strip() != ""]
        if len(devices) < num_workers:
            print(f"[Runner][WARN] devices={len(devices)} < num_workers={num_workers}, will round-robin reuse")
        print(f"[Runner] Mode: multi-process GPUs={devices} workers={num_workers}")
    else:
        devices = [""] * num_workers
        print(f"[Runner] Mode: multi-process (single GPU/CPU) workers={num_workers} device=inherit")

    # 均匀切分 pairs -> shards
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(pairs):
        shards[i % num_workers].append(item)

    counts = [len(s) for s in shards]
    print("[Runner] Shards: " + ", ".join([f"#{i}:{c}" for i, c in enumerate(counts)]))

    q = Queue()
    procs: List[Process] = []
    t0 = time.time()

    # 启动子进程
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

    # 主进程负责合并结果 + 显示整体进度
    while done_workers < num_workers:
        msg = q.get()
        mtype = msg.get("type", "")
        if mtype == "tick":
            pbar.update(1)
        elif mtype == "done":
            cls_results_all.extend(msg.get("results", []))
            done_workers += 1
    pbar.close()

    # 等子进程退出
    for p in procs:
        p.join()

    # 汇总
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
