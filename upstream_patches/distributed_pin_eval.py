import habitat
from tqdm import tqdm
from typing import Dict, Optional
import argparse
import gc
import torch
import os
import sys
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# Run from inside a PIN checkout
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("cwd:", os.getcwd())
print("Python search path:")
for p in sys.path:
    print(p)

from utils.distributed_env import DistributedEnv
from utils.new_top_down_map import *
from utils.goal_detection_metrics import get_goal_seen_metrics, get_found_goal_metrics
from models.navigator.mod_iin_cow_navigator import CowAgent
from habitat.envs.habitat_pin_env import HabitatPINEnv
from utils.wandb_logger import PINDistributedWandbLogger

def evaluate(config_env, args, num_episodes: Optional[int] = None) -> Dict[str, float]:
    exp_name = args.exp_name
    distributed_env = DistributedEnv(config=config_env.habitat, num_jobs=args.num_jobs, job_index=args.job_index)
    wandb_logger = PINDistributedWandbLogger(distributed_env.original_num_episodes, args.num_jobs, args.job_index, tmp_dir=f"output_dir/{exp_name}", debug=args.debug)

    env = HabitatPINEnv(distributed_env, config=config_env)
    agent = CowAgent(config_env)

    if args.save_video:
        # video output directory
        video_dir = os.path.join(os.environ.get("PIN_OUTPUT_DIR", "./output_dir") + f"/{exp_name}", "videos")
        os.makedirs(video_dir, exist_ok=True)

    if num_episodes is None:
        num_episodes = len(env.habitat_env.episodes)
    else:
        assert num_episodes <= len(env.habitat_env.episodes), (
            "num_episodes({}) is larger than number of episodes in environment ({})".format(num_episodes, len(env.habitat_env.episodes))
        )
    assert num_episodes > 0, "num_episodes should be greater than 0"

    count_episodes = 0
    with tqdm(total=num_episodes) as pbar:
        while count_episodes < num_episodes:
            observations = env.reset()
            agent.reset()

            pbar.update(1)
            steps = 0
            num_matched = 0
            goal_seen, goal_seen_cond, found_goal, correct_goal_detected = 0.0, 0.0, 0.0, 0.0
            correct_matches_list, category_matches_list, generic_distractor_matches_list, other_matches_list = [], [], [], []
            matched = False
            frames = []

            with tqdm(total=config_env.habitat.environment.max_episode_steps) as pbar_episode:
                while not env.episode_over:
                    goal_seen, goal_seen_cond = get_goal_seen_metrics(observations, goal_seen, goal_seen_cond)
                    action, matched = agent.act(observations, env)

                    if args.save_video and "rgb" in observations:
                        frame = observations["rgb"]
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        frames.append(frame)

                    found_goal, correct_matches_list, category_matches_list, generic_distractor_matches_list, other_matches_list, num_matched = get_found_goal_metrics(
                        observations, matched, agent.found_goal[0].item(), found_goal, correct_matches_list,
                        category_matches_list, generic_distractor_matches_list, other_matches_list, num_matched
                    )

                    env.apply_action(action)
                    observations = env.get_observation()
                    steps += 1
                    pbar_episode.update(1)

            ep = env.habitat_env.current_episode

            metrics = {
                k: v for k, v in env.habitat_env.get_metrics().items()
                if k not in ["top_down_map", "collisions"]
            }
            metrics["found_goal"] = found_goal
            metrics["goal_seen"] = goal_seen
            metrics["goal_seen_cond"] = goal_seen_cond
            metrics["correct_matches"] = correct_matches_list
            metrics["category_matches"] = category_matches_list
            metrics["generic_distractor_matches"] = generic_distractor_matches_list
            metrics["other_matches"] = other_matches_list
            metrics["num_matched"] = num_matched
            metrics["category"] = ep.object_category
            metrics["scene_id"] = ep.scene_id
            metrics["episode_id"] = ep.episode_id
            metrics["start_position"] = ep.start_position
            metrics["goal_position"] = ep.goals[0].position if len(ep.goals) > 0 else None

            wandb_logger.log(metrics)

            if args.save_video and len(frames) > 0:
                scene_name = os.path.basename(ep.scene_id).replace(".glb", "")
                ep_id = ep.episode_id
                video_path = os.path.join(video_dir, f"{scene_name}_{ep_id}.mp4")
                imageio.mimsave(video_path, frames, fps=10)

            count_episodes += 1

    wandb_logger.close(project="pin", entity='pin', config=dict(config_env), name=exp_name)

def main(args):
    config = habitat.get_config(args.config, args.opts)
    print(config)
    evaluate(config_env=config, args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/pin/pin_hm3d_rgb_v1_cow.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_video", action="store_true", default=True)  # on by default
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--dump_location", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
