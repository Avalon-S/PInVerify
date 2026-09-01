import os
import json
import gzip
import glob
import random
from collections import defaultdict
import argparse
from tqdm import tqdm

def extract_match_pool(dataset_dir, good_episodes_file, output_file):
    """
    Build the match pool from the original dataset and the good-episode list.
    Match pool layout:
    {
        "scene_id": {
            "starts": [ [x, y, z], ... ],
            "goals": [ {position: [x,y,z], object_id: "...", object_category: "..."}, ... ]
        }
    }
    """
    print(f"Loading good episodes list from: {good_episodes_file}")
    with open(good_episodes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "episodes" in data:
            good_eps_list = data["episodes"]
        else:
            good_eps_list = data
    
    # lookup: scene_id -> set of episode ids
    good_eps_lookup = defaultdict(set)
    for ep in good_eps_list:
        # scene_id looks like "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
        # and the dataset file is "00800-TEEsavR23oF.json.gz",
        # so the parent directory name is the key
        try:
            # take the parent directory name
            scene_key = os.path.basename(os.path.dirname(ep["scene_id"]))
            # empty or unexpected, leave it alone
            if not scene_key or scene_key == ".":
                 scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
        except:
             scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
             
        good_eps_lookup[scene_key].add(ep["episode_id"])
    
    print(f"Index built. Found {len(good_eps_list)} good episodes across {len(good_eps_lookup)} scenes.")

    # scan the dataset files
    pattern = os.path.join(dataset_dir, "*.json.gz")
    dataset_files = sorted(glob.glob(pattern))
    
    match_pool = {}
    total_starts = 0
    total_goals = 0

    print(f"Scanning {len(dataset_files)} dataset files in {dataset_dir}...")
    
    for fpath in tqdm(dataset_files):
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
                
            episodes = data.get("episodes", [])
            if not episodes:
                continue
                
            # scene_id from the first episode, or inferred from the filename
            # filenames are {scene_id}.json.gz
            file_scene_key = os.path.basename(fpath).replace(".json.gz", "")
            
            # the episode content is the safer source
            # sample_ep = episodes[0]
            # sample_scene_key = os.path.basename(sample_ep["scene_id"])
            
            # the filename is the key, being unique
            scene_key = file_scene_key
            
            # skip scenes with no good episodes
            if scene_key not in good_eps_lookup:
                continue
                
            good_ids = good_eps_lookup[scene_key]
            
            # store the full pairing: start plus goal
            scene_pairs = []
            
            for idx, ep in enumerate(episodes):
                # Habitat uses a load-order index as episode_id at runtime,
                # so match on the index too
                ep_index_str = str(idx)
                if ep_index_str in good_ids:
                    # a good episode; take the whole pairing
                    start_pos = ep["start_position"]
                    
                    # the first goal; PIN episodes have one
                    if ep["goals"]:
                        # copy the goal wholesale, keeping radius, room_id, room_name and the rest
                        goal_info = ep["goals"][0]
                        
                        # store the pairing
                        pair = {
                            "start_position": start_pos,
                            "start_rotation": ep.get("start_rotation"),
                            "geodesic_distance": ep.get("info", {}).get("geodesic_distance"),
                            "goal": goal_info,
                            "distractors": ep.get("distractors", [])
                        }
                        scene_pairs.append(pair)
            
            if scene_pairs:
                match_pool[scene_key] = {
                    "pairs": scene_pairs  # a list of pairings
                }
                total_starts += len(scene_pairs)
                total_goals += len(scene_pairs)
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

    print(f"Match pool extraction complete.")
    print(f"  Scenes with pool: {len(match_pool)}")
    print(f"  Total valid starts: {total_starts}")
    print(f"  Total valid goals: {total_goals}")
    
    # write the result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(match_pool, f, indent=2)
    print(f"Match pool saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to original dataset content dir (json.gz)")
    parser.add_argument("--good_episodes", type=str, required=True, help="Path to good_episodes.json")
    parser.add_argument("--output", type=str, required=True, help="Path to save match_pool.json")
    args = parser.parse_args()
    
    extract_match_pool(args.dataset_dir, args.good_episodes, args.output)
