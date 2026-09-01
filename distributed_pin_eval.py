#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import jsonlines
import habitat
from tqdm import tqdm
import os
from collections import defaultdict
from typing import Dict, Optional
import imageio
import numpy as np
import quaternion as nq

from habitat_sim.utils.common import quat_from_two_vectors

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# Run from inside a PIN checkout; this makes its modules importable
# no matter where the repository lives.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("cwd:", os.getcwd())
print("Python search path:")
for p in sys.path:
    print(p)

from utils.distributed_env import DistributedEnv
from habitat.envs.habitat_pin_env import HabitatPINEnv
from utils.new_top_down_map import *  # unused here, kept for downstream extensions
from utils.oracle_navigators import ShortestPathFollowerAgentPIN
from utils.wandb_logger import PINDistributedWandbLogger


def evaluate(config_env, args, num_episodes: Optional[int] = None) -> Dict[str, float]:
    """
    Sharded oracle navigation evaluation:
    - DistributedEnv splits the episodes across workers
    - HabitatPINEnv plus a shortest-path follower stands in for the upper bound
    - near the goal the agent turns to face it, tilts, and grabs a snapshot
    - one jsonl per worker, written episode by episode, with optional video export
    - distributed wandb logging through the custom logger
    """
    # === shard the episodes ===
    distributed_env = DistributedEnv(
        config=config_env.habitat,
        num_jobs=args.num_jobs,
        job_index=args.job_index
    )

    # the project's PINEnv wrapper, matching the reference script
    env = HabitatPINEnv(distributed_env, config=config_env)

    # oracle agent, driven straight off the inner habitat_env sim
    agent = ShortestPathFollowerAgentPIN(env.habitat_env.sim, config_env)

    # counts are relative to this shard's episode count
    if num_episodes is None:
        num_episodes = len(env.habitat_env.episodes)
    else:
        assert num_episodes <= len(env.habitat_env.episodes), (
            f"num_episodes({num_episodes}) > available({len(env.habitat_env.episodes)})"
        )
    assert num_episodes > 0, "num_episodes should be greater than 0"

    split = config_env.habitat.dataset.split
    # results root, keyed by split and exp_name (or a timestamp)
    results_name = args.exp_name or datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.results_dir:
        root_dir = os.path.join(args.results_dir, results_name)
    else:
        root_dir = os.path.join("results", "pin", split, results_name)
    os.makedirs(root_dir, exist_ok=True)

    # one jsonl per worker, so concurrent writes never collide
    results_file = os.path.join(root_dir, f"{results_name}_j{args.job_index}_results.jsonl")
    video_dir = os.path.join(root_dir, "videos")
    snapshot_dir = os.path.join(root_dir, "snapshots")

    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)
    if args.save_snapshot:
        os.makedirs(snapshot_dir, exist_ok=True)

    # distributed wandb logger
    wandb_logger = PINDistributedWandbLogger(
        original_num_episodes=distributed_env.original_num_episodes,  # episode count of the full set
        num_jobs=args.num_jobs,
        job_index=args.job_index,
        tmp_dir=root_dir,
        debug=args.debug,
    )

    agg_metrics: Dict = defaultdict(float)
    count_episodes = 0

    with tqdm(total=num_episodes, desc=f"Worker {args.job_index}/{args.num_jobs}") as pbar:
        while count_episodes < num_episodes:
            # HabitatPINEnv exposes reset / get_observation / apply_action / episode_over
            observations = env.reset()
            agent.reset()

            pbar.update(1)
            steps = 0
            frames = []
            snapshot_saved = False

            # step until the episode ends
            while not env.episode_over:
                # === near the goal: face it, tilt, take a snapshot ===
                # reach into habitat_env for the native sim API
                current_state = env.habitat_env.sim.get_agent_state()
                current_pos = np.array(current_state.position, dtype=np.float32)

                # position of the episode's first goal
                ep = env.habitat_env.current_episode
                goal_pos = np.array(ep.goals[0].position, dtype=np.float32) if len(ep.goals) > 0 else None

                if args.save_snapshot and goal_pos is not None and not snapshot_saved:
                    # geodesic distance
                    dist_to_goal = env.habitat_env.sim.geodesic_distance(current_pos, goal_pos)

                    if dist_to_goal is not None and dist_to_goal < 2.0:
                        # turn the agent toward the goal in the horizontal plane
                        direction = goal_pos - current_pos
                        direction[1] = 0.0
                        norm = np.linalg.norm(direction)
                        if norm > 1e-6:
                            direction /= norm
                            # the default forward vector is [0, 0, -1], the Habitat camera forward
                            quat = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)
                            # normalize the quaternion, otherwise habitat_sim rejects it
                            qn = float(np.sqrt(quat.w*quat.w + quat.x*quat.x + quat.y*quat.y + quat.z*quat.z))
                            if qn > 1e-12:
                                quat = nq.quaternion(quat.w/qn, quat.x/qn, quat.y/qn, quat.z/qn)
                            env.habitat_env.sim.set_agent_state(position=current_pos, rotation=quat)

                        # tilt by the height difference between camera and goal
                        try:
                            sensor_offset_y = float(config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1])
                        except Exception:
                            sensor_offset_y = float(config_env.habitat.simulator.agents.agent_0.sim_sensors.rgb_sensor.position[1])
                        # camera world Y = agent foot Y + sensor mount offset
                        cam_y_world = float(current_pos[1]) + sensor_offset_y
                        goal_y = float(goal_pos[1])
                        delta_y = goal_y - cam_y_world
                        tilt_threshold = 0.3
                        if delta_y > tilt_threshold:
                            env.apply_action({"action": "look_up"})
                            observations = env.get_observation()
                        elif delta_y < -tilt_threshold:
                            env.apply_action({"action": "look_down"})
                            observations = env.get_observation()

                        # snapshot; swap in the next observations["rgb"] frame if resolution must match
                        snapshot = env.habitat_env.sim.render(mode="rgb")
                        scene = os.path.basename(ep.scene_id).replace(".glb", "")
                        ep_id = ep.episode_id
                        snapshot_path = os.path.join(snapshot_dir, f"{scene}_{ep_id}_goalview_j{args.job_index}.png")
                        imageio.imwrite(snapshot_path, snapshot)
                        snapshot_saved = True

                # one oracle step
                action = agent.act(observations, env.habitat_env)

                # optional: collect a video frame from the rgb sensor
                if args.save_video and "rgb" in observations:
                    frame = observations["rgb"]
                    # ensure uint8 with shape (H, W, 3)
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8) if frame.dtype in (np.float16, np.float32, np.float64) else frame.astype(np.uint8)
                    frames.append(frame)

                # advance the environment
                env.apply_action(action)
                observations = env.get_observation()
                steps += 1

            # === episode finished: collect metrics ===
            metrics = {
                k: v for k, v in env.habitat_env.get_metrics().items()
                if k not in ["top_down_map"]  # keep the huge map out of the jsonl
            }

            ep = env.habitat_env.current_episode
            metrics["scene_id"] = ep.scene_id
            metrics["episode_id"] = ep.episode_id
            metrics["start_position"] = ep.start_position
            metrics["goal_position"] = ep.goals[0].position if len(ep.goals) > 0 else None

            # record the final pose; pipeline.filter_good_episodes filters on its height
            final_state = env.habitat_env.sim.get_agent_state()
            metrics["final_position"] = [float(x) for x in final_state.position]
            metrics["final_rotation"] = {
                "w": float(final_state.rotation.w),
                "x": float(final_state.rotation.x),
                "y": float(final_state.rotation.y),
                "z": float(final_state.rotation.z),
            }

            # cat_spl may be absent
            spl = float(metrics.get("spl", 0.0))
            cat_spl = float(metrics.get("cat_spl", 0.0))
            pbar.set_description(
                f"W{args.job_index} ep{count_episodes}: len:{metrics.get('episode_length', 0)}, s:{metrics.get('success', 0)}, spl:{round(spl,2)}, cat_spl:{round(cat_spl,2)}"
            )

            # append to this worker's own jsonl
            with jsonlines.open(results_file, mode="a") as f:
                f.write(metrics)

            # distributed wandb record
            wandb_logger.log(metrics)

            # write the video under this worker's own name
            if args.save_video and len(frames) > 0:
                scene = os.path.basename(ep.scene_id).replace(".glb", "")
                ep_id = ep.episode_id
                video_path = os.path.join(video_dir, f"{scene}_{ep_id}_j{args.job_index}.mp4")
                imageio.mimsave(video_path, frames, fps=10)

            # running mean, only for this worker's own console output
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            agg_metrics[m + "/" + str(sub_m)] += sub_v
                elif isinstance(v, (int, float)):
                    agg_metrics[m] += v

            count_episodes += 1

    avg_metrics = {k: v / max(count_episodes, 1) for k, v in agg_metrics.items()}
    print(f"[Worker {args.job_index}] Averages:", avg_metrics)

    wandb_logger.close(
        project="pin",
        entity="pin",
        config=dict(config_env),
        name=f"{results_name}_j{args.job_index}"
    )

    return avg_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/pin/pin_hm3d_v1.yaml")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_snapshot", action="store_true", default=False)
    # sharding arguments, matching the reference script
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Absolute base dir for results (default: results/pin/<split>)")

    # pass --opts through to habitat
    args, unknown = parser.parse_known_args()
    args.opts = [o for o in unknown if "=" in o]
    return args


def main():
    args = parse_args()
    config = habitat.get_config(args.config, args.opts)
    print(config)
    evaluate(config_env=config, args=args)


if __name__ == "__main__":
    main()
