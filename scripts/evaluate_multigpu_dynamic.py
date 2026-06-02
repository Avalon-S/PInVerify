#!/usr/bin/env python3
"""
Multi-GPU Parallel Evaluation with Dynamic Work Stealing

Key improvements over evaluate_multigpu.py:
1. Dynamic task queue - GPUs pull episodes as they complete
2. Better load balancing - fast GPUs don't wait for slow ones
3. No need for FJSP - simple work-stealing queue

Usage:
    python scripts/evaluate_multigpu_dynamic.py --config configs/agent/multi_view_attr_llm.yaml \
        --num_gpus 4 \
        --gpu_ids 0,1,2,3 \
        --base_ports 12182,12282,12382,12482
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager


def load_dataset(index_file: str) -> List[Dict]:
    """Load all episodes from index file."""
    dataset = []
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def worker_process(worker_id: int,
                   gpu_id: int,
                   task_queue: Queue,
                   result_queue: Queue,
                   config_path: str,
                   qwen_port: int,
                   gdino_port: int,
                   output_dir: str,
                   extra_args: List[str],
                   log_file: str):
    """
    Worker process that pulls episodes from shared queue.

    Args:
        worker_id: Worker identifier (0, 1, 2, 3)
        gpu_id: CUDA device ID
        task_queue: Shared queue of episode indices
        result_queue: Queue for collecting results
        config_path: Path to agent config YAML
        qwen_port: Qwen server port
        gdino_port: GDINO server port
        output_dir: Output directory for this worker
        extra_args: Additional CLI args for evaluate.py
        log_file: Log file path
    """
    import subprocess

    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Worker {worker_id} / GPU {gpu_id}] Started")
    print(f"[Worker {worker_id}] Ports: Qwen={qwen_port}, GDINO={gdino_port}")
    print(f"[Worker {worker_id}] Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open log file
    log_f = open(log_file, 'w', buffering=1)
    log_f.write(f"Worker {worker_id} started on GPU {gpu_id}\n")
    log_f.write(f"Qwen port: {qwen_port}, GDINO port: {gdino_port}\n")
    log_f.write("="*60 + "\n\n")

    episodes_processed = 0

    try:
        while True:
            # Get next task from queue (blocking)
            try:
                episode_idx = task_queue.get(timeout=5)  # 5s timeout
            except:
                # Queue is empty and no more tasks
                break

            if episode_idx is None:
                # Poison pill - shutdown signal
                break

            log_f.write(f"[{time.strftime('%H:%M:%S')}] Processing episode {episode_idx}\n")
            log_f.flush()

            # Build command for single episode
            cmd = [
                "python", "scripts/evaluate.py",
                "--config", config_path,
                f"+start_idx={episode_idx}",
                f"+end_idx={episode_idx + 1}",
                f"server.qwen_text_url=http://127.0.0.1:{qwen_port}/qwen-text",
                f"server.qwen_vl_url=http://127.0.0.1:{qwen_port}/qwen-vl",
                f"server.gdino_url=http://127.0.0.1:{gdino_port}/groundingdino",
                f"output.root={output_dir}",
            ]
            cmd.extend(extra_args)

            # Run single episode
            start_time = time.time()
            try:
                # Write command to log
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write("-" * 60 + "\n")
                log_f.flush()

                # Run with output redirected to log file
                result = subprocess.run(cmd, env=env,
                                       stdout=log_f,
                                       stderr=subprocess.STDOUT,
                                       check=True)

                elapsed = time.time() - start_time
                log_f.write("\n" + "-" * 60 + "\n")
                log_f.write(f"[{time.strftime('%H:%M:%S')}] Episode {episode_idx} completed in {elapsed:.1f}s\n")
                log_f.flush()

                episodes_processed += 1

                # Report success
                result_queue.put({
                    'worker_id': worker_id,
                    'episode_idx': episode_idx,
                    'status': 'success',
                    'elapsed': elapsed
                })

            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start_time
                log_f.write("\n" + "-" * 60 + "\n")
                log_f.write(f"[{time.strftime('%H:%M:%S')}] Episode {episode_idx} FAILED after {elapsed:.1f}s\n")
                log_f.write(f"Exit code: {e.returncode}\n")
                log_f.flush()

                # Report failure
                result_queue.put({
                    'worker_id': worker_id,
                    'episode_idx': episode_idx,
                    'status': 'failed',
                    'elapsed': elapsed,
                    'error': str(e)
                })

    finally:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Worker {worker_id} finished: {episodes_processed} episodes processed\n")
        log_f.close()

        # Signal completion
        result_queue.put({
            'worker_id': worker_id,
            'status': 'worker_done',
            'episodes_processed': episodes_processed
        })


def merge_results(output_dir: str, num_workers: int, cfg=None):
    """
    Merge results from all worker subdirectories.

    Each worker saves to {output_dir}/worker_{worker_id}/results.json
    Final merged results go to {output_dir}/results.json
    """
    from pver.eval.metrics import calculate_metrics

    print(f"\n{'='*60}")
    print("Merging results from all workers...")
    print(f"{'='*60}\n")

    all_results = []

    for worker_id in range(num_workers):
        worker_dir = os.path.join(output_dir, f"worker_{worker_id}")
        results_file = os.path.join(worker_dir, "results.json")

        if not os.path.exists(results_file):
            print(f"[WARN] Results file not found: {results_file}")
            continue

        with open(results_file, 'r', encoding='utf-8') as f:
            worker_results = json.load(f)

        all_results.extend(worker_results)
        print(f"Loaded {len(worker_results)} episodes from worker {worker_id}")

    # Deduplicate: same episode may appear if output dir had stale data
    seen_keys = {}
    deduped = []
    for ep in all_results:
        key = (ep.get("scene_id", ""),
               ep.get("episode_id", ""),
               ep.get("target_object_id", ""),
               ep.get("object_id", ""),
               ep.get("pair_type", ""))
        if key not in seen_keys:
            seen_keys[key] = len(deduped)
            deduped.append(ep)
        else:
            # Keep the later one (more likely from the current run)
            deduped[seen_keys[key]] = ep

    n_dupes = len(all_results) - len(deduped)
    all_results = deduped

    # Save merged results
    final_results_file = os.path.join(output_dir, "results.json")
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if n_dupes:
        print(f"\n[WARN] Removed {n_dupes} duplicate episodes (stale data in worker dirs)")
    print(f"Merged {len(all_results)} unique episodes")

    # Calculate metrics
    metrics = calculate_metrics(all_results)

    # Inject config info if available
    if cfg is not None:
        metrics["config_info"] = {
            "agent_type": cfg.method.get("agent_type", "unknown"),
            "nbv_type": cfg.method.get("nbv", {}).get("type", "none"),
            "fusion_type": cfg.method.get("fusion", {}).get("type", "none"),
            "bbox_mode": cfg.method.get("bbox_mode", "unknown"),
            "max_steps": cfg.env.get("max_steps", 1),
            "query_mode": cfg.method.get("query_mode", "unknown"),
        }

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Pretty print summary (clean, no episode lists)
    print("\n" + "="*60)
    print("                    EVALUATION SUMMARY")
    print("="*60)
    print(f"\nOverall Results:")
    print(f"   Total Episodes: {metrics['total_episodes']}")
    print(f"   Accuracy:       {metrics['accuracy']:.2%}")
    print(f"   Correct:        {metrics['correct_count']}")
    print(f"   Wrong:          {metrics['wrong_count']}")
    print(f"   Avg Steps:      {metrics['asd']:.2f}")

    print(f"\nPer Pair Type:")
    print("-"*60)
    for pt, data in metrics.get('per_pair_type', {}).items():
        print(f"   [{pt.upper()}]")
        print(f"      Accuracy: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    nav_stats = metrics.get('nav_stats', {})
    if nav_stats.get('total_nav_failures', 0) > 0:
        print(f"\nNavigation Failures:")
        print("-"*60)
        print(f"   Total Nav Failures:    {nav_stats['total_nav_failures']}")
        print(f"     - Unreachable:       {nav_stats.get('nav_fail_unreachable', 0)}")
        print(f"     - Trap Views:        {nav_stats.get('nav_fail_trap', 0)}")
        print(f"   Episodes Affected:     {nav_stats['episodes_with_nav_failure']}/{metrics['total_episodes']} ({nav_stats['nav_failure_rate_per_episode']:.1%})")
        print(f"   Failure Rate Per Step: {nav_stats['nav_failure_rate_per_step']:.1%}")
        print(f"   Avg Failures/Episode:  {nav_stats['avg_nav_failures_per_episode']:.2f}")

    diag = metrics.get('diagnostic_stats', {})
    if diag:
        print(f"\nDiagnostic Stats:")
        print("-"*60)
        fv_acc = diag.get('first_view_accuracy')
        if fv_acc is not None:
            print(f"   First-View Accuracy:   {fv_acc:.2%}")
        print(f"   Avg Effective Views:   {diag.get('avg_effective_views', 0):.2f}")

    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Dynamic Work Stealing Evaluation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs")
    parser.add_argument("--base_ports", type=str, required=True,
                       help="Comma-separated Qwen ports (e.g., '12182,12282,12382,12482')")
    parser.add_argument("--port_offset", type=int, default=1,
                       help="GDINO port = base_port + offset")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory")
    parser.add_argument("--dataset_index", type=str, default=None,
                       help="Override dataset index file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from completed episodes (skip existing episode.json)")

    args, extra_args = parser.parse_known_args()

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip()]
    else:
        gpu_ids = list(range(args.num_gpus))

    if len(gpu_ids) != args.num_gpus:
        print(f"Error: num_gpus={args.num_gpus} but {len(gpu_ids)} GPU IDs provided")
        return 1

    # Parse ports
    base_ports = [int(x) for x in args.base_ports.split(',')]
    if len(base_ports) != args.num_gpus:
        print(f"Error: num_gpus={args.num_gpus} but {len(base_ports)} ports provided")
        return 1

    # Load config
    import yaml
    from omegaconf import OmegaConf

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = OmegaConf.load(f)


    # Determine index file
    if args.dataset_index:
        index_file = args.dataset_index
    else:
        index_file = cfg.dataset.index_file

    # Handle relative paths - try to construct full path from dataset root
    if not os.path.isabs(index_file):
        if not os.path.exists(index_file):
            # Try dataset.root/val/index_file first
            index_candidate = os.path.join(cfg.dataset.root, "val", index_file)
            if os.path.exists(index_candidate):
                index_file = index_candidate
            else:
                # Try dataset.root/index_file
                index_candidate = os.path.join(cfg.dataset.root, index_file)
                if os.path.exists(index_candidate):
                    index_file = index_candidate

    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        print(f"Tried paths:")
        print(f"  - {args.dataset_index if args.dataset_index else cfg.dataset.index_file}")
        if not os.path.isabs(args.dataset_index if args.dataset_index else cfg.dataset.index_file):
            print(f"  - {os.path.join(cfg.dataset.root, 'val', os.path.basename(index_file))}")
            print(f"  - {os.path.join(cfg.dataset.root, os.path.basename(index_file))}")
        return 1

    # Load dataset
    dataset = load_dataset(index_file)
    total_episodes = len(dataset)

    print(f"\n{'='*60}")
    print("Multi-GPU Dynamic Work Stealing Evaluation")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Dataset: {index_file}")
    print(f"Episodes: {total_episodes}")
    print(f"GPUs: {args.num_gpus} ({gpu_ids})")
    print(f"Base Ports: {base_ports}")
    print(f"Strategy: Dynamic task queue (work stealing)")
    if args.resume:
        print(f"Resume: enabled")
    print(f"{'='*60}\n")

    # Determine output directory (needed for resume scan)
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = cfg.output.root

    # ---- Resume: scan for completed episodes ----
    completed_indices = set()
    if args.resume and os.path.exists(output_base):
        print("Resume: scanning for completed episodes...")

        def _episode_key(d_item):
            pt = d_item.get("pair_type", "positive")
            ep = str(d_item.get("episode_id") or d_item.get("episode", ""))
            sc = d_item.get("scene_key") or d_item.get("scene")
            if not sc:
                ep_path = d_item.get("episode_path", "")
                path_parts = ep_path.split("/")
                if len(path_parts) >= 3:
                    sc = path_parts[-2]
            sc = sc or "unknown_scene"
            return (pt, sc, ep)

        # Scan all worker directories for completed episode.json files
        completed_keys = set()
        for worker_id in range(args.num_gpus):
            worker_dir = os.path.join(output_base, f"worker_{worker_id}")
            if not os.path.exists(worker_dir):
                continue
            for dirpath, _dirnames, filenames in os.walk(worker_dir):
                if "episode.json" in filenames:
                    rel = os.path.relpath(dirpath, worker_dir)
                    parts = rel.replace("\\", "/").split("/")
                    # Structure: pair_type / correct|wrong / scene / ep_id
                    if len(parts) >= 4:
                        completed_keys.add((parts[0], parts[2], parts[3]))

        # Match against dataset to find completed indices
        for idx, item in enumerate(dataset):
            if _episode_key(item) in completed_keys:
                completed_indices.add(idx)

        print(f"Resume: found {len(completed_indices)} completed episodes, skipping them")
        print(f"Resume: {total_episodes - len(completed_indices)} episodes remaining\n")

    # Create task queue (skip completed episodes in resume mode)
    task_queue = Queue()
    enqueued = 0
    for idx in range(total_episodes):
        if idx not in completed_indices:
            task_queue.put(idx)
            enqueued += 1

    # Add poison pills for workers to know when to stop
    for _ in range(args.num_gpus):
        task_queue.put(None)

    # Create result queue
    result_queue = Queue()

    # Create log directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/multigpu_dynamic/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Create symlink
    latest_link = "logs/multigpu_dynamic/latest"
    if os.path.exists(latest_link) or os.path.islink(latest_link):
        os.remove(latest_link)
    os.symlink(timestamp, latest_link, target_is_directory=True)

    print(f"Log directory: {log_dir}\n")

    # Pass resume to evaluate.py subprocesses too
    worker_extra_args = list(extra_args)
    if args.resume:
        worker_extra_args.append("+resume=true")

    # Launch workers
    workers = []
    for worker_id in range(args.num_gpus):
        gpu_id = gpu_ids[worker_id]
        base_port = base_ports[worker_id]

        qwen_port = base_port
        gdino_port = base_port + args.port_offset

        worker_output_dir = os.path.join(output_base, f"worker_{worker_id}")
        log_file = os.path.join(log_dir, f"worker_{worker_id}.log")

        p = Process(target=worker_process, args=(
            worker_id,
            gpu_id,
            task_queue,
            result_queue,
            args.config,
            qwen_port,
            gdino_port,
            worker_output_dir,
            worker_extra_args,
            log_file
        ))
        p.start()
        workers.append(p)

    print(f"Launched {args.num_gpus} workers\n")

    # Monitor progress
    resumed_count = len(completed_indices)
    completed_episodes = 0
    failed_episodes = 0
    workers_done = 0

    start_time = time.time()

    print("Progress:")
    if resumed_count > 0:
        print(f"  (Resumed {resumed_count} episodes, running {enqueued} remaining)")
    print("-" * 60)

    while workers_done < args.num_gpus:
        try:
            result = result_queue.get(timeout=10)

            if result['status'] == 'worker_done':
                workers_done += 1
                # Print on new line for worker completion
                print(f"\n[Worker {result['worker_id']}] Finished ({result['episodes_processed']} episodes)")
            elif result['status'] == 'success':
                completed_episodes += 1
                done_total = resumed_count + completed_episodes + failed_episodes
                progress = done_total / total_episodes * 100
                # Update progress on same line
                print(f"\r[{progress:5.1f}%] Completed: {resumed_count + completed_episodes}/{total_episodes} | Failed: {failed_episodes} | Latest: ep{result['episode_idx']} by worker{result['worker_id']} ({result['elapsed']:.1f}s)", end='', flush=True)
            elif result['status'] == 'failed':
                failed_episodes += 1
                done_total = resumed_count + completed_episodes + failed_episodes
                progress = done_total / total_episodes * 100
                # Print failures on new line (important events)
                print(f"\n[{progress:5.1f}%] Episode {result['episode_idx']} FAILED on worker {result['worker_id']}")
        except:
            # Timeout - check if workers are still alive
            pass

    # Wait for all workers to finish
    for p in workers:
        p.join()

    elapsed_total = time.time() - start_time

    print("\n" + "-" * 60)
    print(f"All workers finished in {elapsed_total:.1f}s")
    print(f"Completed: {completed_episodes}, Failed: {failed_episodes}")

    # Merge results
    merge_results(output_base, args.num_gpus, cfg=cfg)

    return 0


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
