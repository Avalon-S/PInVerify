import gzip
import json
import os
import glob
from tqdm import tqdm

dataset_dir = "data/datasets/pin/hm3d/v1/val/content"
bad_episodes_file = "results/val/eval_pin_goalview/bad_episodes.json"

print(f"Scanning dataset in: {dataset_dir}")

files = sorted(glob.glob(os.path.join(dataset_dir, "*.json.gz")))
print(f"Found {len(files)} files.")

total_episodes = 0
all_episode_ids = set()

for fpath in tqdm(files):
    try:
        scene_key = os.path.basename(fpath).replace(".json.gz", "")
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            data = json.load(f)
            episodes = data.get("episodes", [])
            total_episodes += len(episodes)
            for ep in episodes:
                # record (scene_key, episode_id)
                all_episode_ids.add((scene_key, str(ep["episode_id"])))
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

print(f"\nTotal Actual Episodes in Dataset: {total_episodes}")

# Now check against bad_episodes
print(f"\nChecking against bad episodes list: {bad_episodes_file}")
try:
    with open(bad_episodes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "episodes" in data:
            bad_eps_list = data["episodes"]
        else:
            bad_eps_list = data
            
    print(f"Total Bad Episodes listed: {len(bad_eps_list)}")
    
    ghost_count = 0
    ghost_examples = []
    
    for ep in bad_eps_list:
        # Reconstruct key logic
        try:
            scene_key = os.path.basename(os.path.dirname(ep["scene_id"]))
            if not scene_key or scene_key == ".":
                 scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
        except:
             scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
             
        eid = str(ep["episode_id"])
        
        if (scene_key, eid) not in all_episode_ids:
            ghost_count += 1
            if len(ghost_examples) < 10:
                ghost_examples.append(f"{scene_key} - {eid}")

    print(f"Ghost Bad Episodes (Listed but not in dataset): {ghost_count}")
    if ghost_count > 0:
        print("Examples of ghosts:")
        for g in ghost_examples:
            print(f"  - {g}")
            
except Exception as e:
    print(f"Error processing bad list: {e}")
