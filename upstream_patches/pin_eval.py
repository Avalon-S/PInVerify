# import argparse
# import datetime
# import jsonlines
# import habitat
# from tqdm import tqdm
# import os
# from collections import defaultdict
# from typing import Dict, Optional
# import imageio
# import numpy as np

# from habitat import Env
# from habitat_sim.utils.common import quat_from_two_vectors

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# sys.path is set below, relative to this file

# print("cwd:", os.getcwd())
# print("Python search path:")
# for p in sys.path:
#     print(p)

# from utils.new_top_down_map import *
# from utils.oracle_navigators import ShortestPathFollowerAgentPIN
# from utils.wandb_logger import PINDistributedWandbLogger


# def evaluate(config_env, args, num_episodes: Optional[int] = None) -> Dict[str, float]:
#     split = config_env.habitat.dataset.split
#     env = Env(config=config_env)
#     agent = ShortestPathFollowerAgentPIN(env.sim, config_env)

#     if num_episodes is None:
#         num_episodes = len(env.episodes)
#     else:
#         assert num_episodes <= len(env.episodes), (
#             f"num_episodes({num_episodes}) > available({len(env.episodes)})"
#         )
#     assert num_episodes > 0, "num_episodes should be greater than 0"

#     agg_metrics: Dict = defaultdict(float)
#     os.makedirs(f"results/pin/{split}", exist_ok=True)

#     results_name = args.exp_name or datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     results_file = f"results/pin/{split}/{results_name}_results.jsonl"
#     video_dir = f"results/pin/{split}/{results_name}/videos"
#     snapshot_dir = f"results/pin/{split}/{results_name}/snapshots"

#     if args.save_video:
#         os.makedirs(video_dir, exist_ok=True)
#     os.makedirs(snapshot_dir, exist_ok=True)

#     wandb_logger = PINDistributedWandbLogger(
#         original_num_episodes=num_episodes,
#         num_jobs=1,
#         job_index=0,
#         tmp_dir=f"results/pin/{split}/{results_name}",
#         debug=False,
#     )

#     count_episodes = 0
#     with tqdm(total=num_episodes) as pbar:
#         while count_episodes < num_episodes:
#             observations = env.reset()
#             agent.reset()
#             pbar.update(1)
#             steps = 0
#             frames = []
#             snapshot_saved = False

#             while not env.episode_over:
#                 # current agent position and goal position
#                 current_pos = env.sim.get_agent_state().position
#                 goal_pos = env.current_episode.goals[0].position
#                 dist_to_goal = env.sim.geodesic_distance(current_pos, goal_pos)

#                 # close to the goal and no snapshot saved yet
#                 if dist_to_goal < 2.0 and not snapshot_saved:   # distance threshold
#                     direction = np.array(goal_pos) - np.array(current_pos)
#                     direction[1] = 0
#                     direction /= np.linalg.norm(direction)
#                     quat = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)
#                     env.sim.set_agent_state(position=current_pos, rotation=quat)

#                     snapshot = env.sim.render(mode="rgb")
#                     scene = os.path.basename(env.current_episode.scene_id).replace(".glb", "")
#                     ep_id = env.current_episode.episode_id
#                     snapshot_path = os.path.join(snapshot_dir, f"{scene}_{ep_id}_goalview.png")
#                     imageio.imwrite(snapshot_path, snapshot)
#                     snapshot_saved = True

#                 action = agent.act(observations, env)

#                 if args.save_video:
#                     frame = observations["rgb"]
#                     frames.append(frame)

#                 observations = env.step(action)
#                 steps += 1

#             metrics = env.get_metrics()
#             ep = env.current_episode

#             metrics["scene_id"] = ep.scene_id
#             metrics["episode_id"] = ep.episode_id
#             metrics["start_position"] = ep.start_position
#             metrics["goal_position"] = ep.goals[0].position if len(ep.goals) > 0 else None

#             pbar.set_description(
#                 f"{count_episodes}: length:{metrics['episode_length']}, s:{metrics['success']}, spl:{round(metrics['spl'], 2)}, cat_spl:{round(metrics['cat_spl'], 2)}"
#             )

#             with jsonlines.open(results_file, mode="a") as f:
#                 f.write(metrics)

#             wandb_logger.log(metrics)

#             if args.save_video and len(frames) > 0:
#                 scene = os.path.basename(ep.scene_id).replace(".glb", "")
#                 ep_id = ep.episode_id
#                 video_path = os.path.join(video_dir, f"{scene}_{ep_id}.mp4")
#                 imageio.mimsave(video_path, frames, fps=10)

#             for m, v in metrics.items():
#                 if isinstance(v, dict):
#                     for sub_m, sub_v in v.items():
#                         if isinstance(sub_v, (int, float)):
#                             agg_metrics[m + "/" + str(sub_m)] += sub_v
#                 elif isinstance(v, (int, float)):
#                     agg_metrics[m] += v

#             count_episodes += 1

#     avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
#     print(avg_metrics)

#     wandb_logger.close(
#         project="pin", entity="pin", config=dict(config_env), name=results_name
#     )

#     return avg_metrics


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="configs/models/pin/pin_hm3d_v1.yaml")
#     parser.add_argument("--save_video", action="store_true", default=True)
#     parser.add_argument("--save_video", action="store_true")
#     args, unknown = parser.parse_known_args()
#     args.opts = [o for o in unknown if "=" in o]
#     return args


# def main():
#     args = parse_args()
#     config = habitat.get_config(args.config, args.opts)
#     evaluate(config_env=config, args=args)


# if __name__ == "__main__":
#     main()


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
import math

from habitat import Env
from habitat_sim.utils.common import quat_from_two_vectors

# import for the custom action
# from configs.habitat.actions.look_actions import LookUpAction, LookDownAction

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# Run from inside a PIN checkout; this makes its modules importable
# no matter where the repository lives.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("cwd:", os.getcwd())
print("Python search path:")
for p in sys.path:
    print(p)

from utils.new_top_down_map import *
from utils.oracle_navigators import ShortestPathFollowerAgentPIN
from utils.wandb_logger import PINDistributedWandbLogger

def evaluate(config_env, args, num_episodes: Optional[int] = None) -> Dict[str, float]:
    split = config_env.habitat.dataset.split
    env = Env(config=config_env)
    agent = ShortestPathFollowerAgentPIN(env.sim, config_env)

    if num_episodes is None:
        num_episodes = len(env.episodes)
    else:
        assert num_episodes <= len(env.episodes), (
            f"num_episodes({num_episodes}) > available({len(env.episodes)})"
        )
    assert num_episodes > 0, "num_episodes should be greater than 0"

    agg_metrics: Dict = defaultdict(float)
    os.makedirs(f"results/pin/{split}", exist_ok=True)

    results_name = args.exp_name or datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_file = f"results/pin/{split}/{results_name}_results.jsonl"
    video_dir = f"results/pin/{split}/{results_name}/videos"
    snapshot_dir = f"results/pin/{split}/{results_name}/snapshots"

    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    wandb_logger = PINDistributedWandbLogger(
        original_num_episodes=num_episodes,
        num_jobs=1,
        job_index=0,
        tmp_dir=f"results/pin/{split}/{results_name}",
        debug=False,
    )

    count_episodes = 0
    with tqdm(total=num_episodes) as pbar:
        while count_episodes < num_episodes:
            observations = env.reset()
            agent.reset()
            pbar.update(1)
            steps = 0
            frames = []
            snapshot_saved = False

            while not env.episode_over:
                current_pos = env.sim.get_agent_state().position
                goal_pos = env.current_episode.goals[0].position
                dist_to_goal = env.sim.geodesic_distance(current_pos, goal_pos)

                if dist_to_goal < 2.0 and not snapshot_saved:
                    # face the agent toward the goal
                    direction = np.array(goal_pos) - np.array(current_pos)
                    direction[1] = 0
                    direction /= np.linalg.norm(direction)
                    quat = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)
                    env.sim.set_agent_state(position=current_pos, rotation=quat)

                    # decide whether a tilt is needed
                    goal_y = goal_pos[1]
                    camera_y = config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1]
                    delta_y = goal_y - camera_y
                    tilt_threshold = 0.2

                    if delta_y > tilt_threshold:
                        observations = env.step({"action": "LOOK_UP"})
                    elif delta_y < -tilt_threshold:
                        observations = env.step({"action": "LOOK_DOWN"})

                    snapshot = env.sim.render(mode="rgb")

                    scene = os.path.basename(env.current_episode.scene_id).replace(".glb", "")
                    ep_id = env.current_episode.episode_id
                    snapshot_path = os.path.join(snapshot_dir, f"{scene}_{ep_id}_goalview.png")
                    imageio.imwrite(snapshot_path, snapshot)
                    snapshot_saved = True

                action = agent.act(observations, env)

                if args.save_video:
                    frame = observations["rgb"]
                    frames.append(frame)

                observations = env.step(action)
                steps += 1

            metrics = env.get_metrics()
            ep = env.current_episode

            metrics["scene_id"] = ep.scene_id
            metrics["episode_id"] = ep.episode_id
            metrics["start_position"] = ep.start_position
            metrics["goal_position"] = ep.goals[0].position if len(ep.goals) > 0 else None

            pbar.set_description(
                f"{count_episodes}: length:{metrics['episode_length']}, s:{metrics['success']}, spl:{round(metrics['spl'], 2)}, cat_spl:{round(metrics['cat_spl'], 2)}"
            )

            with jsonlines.open(results_file, mode="a") as f:
                f.write(metrics)

            wandb_logger.log(metrics)

            if args.save_video and len(frames) > 0:
                scene = os.path.basename(ep.scene_id).replace(".glb", "")
                ep_id = ep.episode_id
                video_path = os.path.join(video_dir, f"{scene}_{ep_id}.mp4")
                imageio.mimsave(video_path, frames, fps=10)

            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            agg_metrics[m + "/" + str(sub_m)] += sub_v
                elif isinstance(v, (int, float)):
                    agg_metrics[m] += v

            count_episodes += 1

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
    print(avg_metrics)

    wandb_logger.close(
        project="pin", entity="pin", config=dict(config_env), name=results_name
    )

    return avg_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/pin/pin_hm3d_v1.yaml")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=True)
    args, unknown = parser.parse_known_args()
    args.opts = [o for o in unknown if "=" in o]
    return args


def main():
    args = parse_args()
    config = habitat.get_config(args.config, args.opts)
    evaluate(config_env=config, args=args)


if __name__ == "__main__":
    main()
