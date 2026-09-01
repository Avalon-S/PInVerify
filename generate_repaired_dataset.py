import os
import json
import gzip
import glob
import random
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict

def generate_repaired_dataset(original_dataset_dir, bad_episodes_file, match_pool_file, output_dir):
    """
    Write the repaired dataset.
    Each bad episode gets a start position and goal sampled from the match pool.
    """
    print(f"Loading Match Pool from: {match_pool_file}")
    with open(match_pool_file, "r", encoding="utf-8") as f:
        match_pool = json.load(f)
        
    print(f"Loading Bad Episodes list from: {bad_episodes_file}")
    with open(bad_episodes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "episodes" in data:
            bad_eps_list = data["episodes"]
        else:
            bad_eps_list = data
        
    # lookup: scene_key -> set of bad episode ids
    bad_eps_lookup = defaultdict(set)
    for ep in bad_eps_list:
        # must match extract_match_pool.py
        # scene_id: "data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
        # the key is "00800-TEEsavR23oF"
        try:
             # take the parent directory name
            scene_key = os.path.basename(os.path.dirname(ep["scene_id"]))
            if not scene_key or scene_key == ".":
                 scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
        except:
             scene_key = os.path.basename(ep["scene_id"]).split(".")[0]
             
        bad_eps_lookup[scene_key].add(str(ep["episode_id"]))
        
    print(f"Bad episodes indexed. Processing files...")

    pattern = os.path.join(original_dataset_dir, "*.json.gz")
    dataset_files = sorted(glob.glob(pattern))
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_repaired = 0
    repaired_ids_set = set()
    total_skipped = 0 # unrepairable: the pool was empty
    
    for fpath in tqdm(dataset_files):
        filename = os.path.basename(fpath)
        scene_key = filename.replace(".json.gz", "")
        
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
                
            episodes = data.get("episodes", [])
            new_episodes = []
            modified = False
            
            # a file with no bad episodes could be copied verbatim,
            # but it is rewritten anyway for consistency
            
            pool_pairs = []
            if scene_key in match_pool:
                pool_pairs = match_pool[scene_key]["pairs"]
            
            bad_ids_in_scene = bad_eps_lookup.get(scene_key, set())
            
            for idx, ep in enumerate(episodes):
                # Habitat uses a load-order index as episode_id at runtime,
                # so match on the index too
                ep_index_str = str(idx)
                
                if ep_index_str in bad_ids_in_scene:
                    # needs repair
                    if not pool_pairs:
                        # empty pool, leave it as it is
                        new_episodes.append(ep)
                        continue
                    
                    # current start and goal, possibly from an earlier repair round
                    current_start = ep.get("start_position", [])
                    current_goal_pos = None
                    if ep.get("goals") and len(ep["goals"]) > 0:
                        current_goal_pos = ep["goals"][0].get("position", [])
                    
                    # drop the pairing identical to the current one, so a repair actually changes something
                    def positions_equal(pos1, pos2, tol=1e-4):
                        if pos1 is None or pos2 is None:
                            return False
                        if len(pos1) != len(pos2):
                            return False
                        return all(abs(a - b) < tol for a, b in zip(pos1, pos2))
                    
                    candidate_pairs = []
                    for pair in pool_pairs:
                        pair_start = pair["start_position"]
                        pair_goal_pos = pair["goal"]["position"]
                        
                        # same start and same goal means the same pairing
                        start_same = positions_equal(current_start, pair_start)
                        goal_same = positions_equal(current_goal_pos, pair_goal_pos)
                        
                        if start_same and goal_same:
                            continue  # skip it
                        
                        candidate_pairs.append(pair)
                    
                    if not candidate_pairs:
                        # every pairing is used up; no repair possible
                        print(f"  [WARN] Scene '{scene_key}' Episode {ep_index_str}: All pairs exhausted, keeping current.")
                        new_episodes.append(ep)
                        continue
                        
                    # sample from the candidates
                    sampled_pair = random.choice(candidate_pairs)
                    new_start = sampled_pair["start_position"]
                    new_rotation = sampled_pair.get("start_rotation")
                    new_dist = sampled_pair.get("geodesic_distance")
                    new_distractors = sampled_pair.get("distractors", [])
                    new_goal_info = sampled_pair["goal"]
                    
                    # deep copy
                    new_ep = copy.deepcopy(ep)
                    
                    # replace the start
                    new_ep["start_position"] = new_start
                    if new_rotation is not None:
                        new_ep["start_rotation"] = new_rotation
                        
                    # replace the geodesic distance
                    if new_dist is not None:
                        if "info" not in new_ep: new_ep["info"] = {}
                        new_ep["info"]["geodesic_distance"] = new_dist
                    
                    # === replace distractors: geometry from the good episode, identity from the original ===
                    # Source of Geometry: Good Episode (sampled_pair["distractors"])
                    # Source of Identity: Bad Episode (ep["distractors"])
                    good_distractors_geo = sampled_pair.get("distractors", [])
                    original_distractors = ep.get("distractors", [])
                    
                    merged_distractors = []
                    # only min(good positions, original objects) distractors can be placed;
                    # surplus original objects are dropped, having nowhere legal to go,
                    # and surplus good positions go unused
                    num_to_place = min(len(good_distractors_geo), len(original_distractors))
                    
                    for i in range(num_to_place):
                        # geometry, position and room, from the good episode
                        geo_data = good_distractors_geo[i]
                        # identity, category and id, from the original
                        id_data = original_distractors[i]
                        
                        new_d = copy.deepcopy(geo_data) # start from the position data
                        new_d["object_id"] = id_data["object_id"]
                        new_d["object_category"] = id_data["object_category"]
                        if "object_name" in id_data:
                            new_d["object_name"] = id_data["object_name"]
                            
                        merged_distractors.append(new_d)
                        
                    new_ep["distractors"] = merged_distractors
                    
                    # === replace the goal the same way ===
                    # Source of Geometry: Good Episode (sampled_pair["goal"])
                    # Source of Identity: Bad Episode (ep["goals"][0])
                    good_goal_geo = sampled_pair["goal"]
                    original_goal_id = ep["goals"][0]
                    
                    # copy the good position, radius and room_id included
                    new_goal = copy.deepcopy(good_goal_geo)
                    # override the identity
                    new_goal["object_id"] = original_goal_id["object_id"]
                    new_goal["object_category"] = original_goal_id["object_category"]
                    if "object_name" in original_goal_id:
                        new_goal["object_name"] = original_goal_id["object_name"]
                        
                    # it is a list
                    new_ep["goals"] = [new_goal]
                    
                    # update the top-level info
                    if "object_id" in new_goal:
                         new_ep["object_id"] = new_goal["object_id"]
                    if "object_category" in new_goal:
                         new_ep["object_category"] = new_goal["object_category"]
                    
                    # flag it as repaired
                    if "info" not in new_ep: new_ep["info"] = {}
                    new_ep["info"]["repaired"] = True
                    
                    new_episodes.append(new_ep)
                    total_repaired += 1
                    repaired_ids_set.add((scene_key, ep_index_str))
                    modified = True
                else:
                    # no repair needed
                    new_episodes.append(ep)
            
            # write the new file
            data["episodes"] = new_episodes
            output_path = os.path.join(output_dir, filename)
            
            # unchanged files are written too, since this is a new dataset release
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    print(f"Repair generation complete.")
    print(f"Total bad episodes listed: {len(bad_eps_list)}")
    print(f"Total episodes repaired: {total_repaired}")
    
    unrepaired_count_total = len(bad_eps_list) - total_repaired
    print(f"Total unrepaired: {unrepaired_count_total}")
    
    if unrepaired_count_total > 0:
        print("\n--- Detailed Unrepaired List ---")
        count = 0
        for sk, eids in bad_eps_lookup.items():
            for eid in eids:
                if (sk, eid) not in repaired_ids_set:
                    # Check pool status
                    reason = "Unknown"
                    if sk not in match_pool:
                        reason = "No Match Pool for Scene"
                    elif not match_pool[sk]["starts"]:
                        reason = "Match Pool Empty"
                    else:
                        reason = "Episode NOT FOUND in Dataset (Ghost) or Key Mismatch"
                        
                    print(f"  [MISSING] Scene: '{sk}' | Episode: {eid} | Reason: {reason}")
                    count += 1
                    if count > 20:
                         print(f"  ... and {unrepaired_count_total - 20} more.")
                         break
            if count > 20: break

    print(f"\nNew dataset saved to: {output_dir}")
            
    print(f"\nNew dataset saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True, help="Original dataset content dir")
    parser.add_argument("--bad_episodes", type=str, required=True, help="bad_episodes.json")
    parser.add_argument("--match_pool", type=str, required=True, help="match_pool.json")
    parser.add_argument("--output_dir", type=str, required=True, help="New dataset content dir")
    args = parser.parse_args()
    
    generate_repaired_dataset(args.original_dir, args.bad_episodes, args.match_pool, args.output_dir)
