import gzip
import json
import os

# the file named in the error being investigated
target_file = "data/datasets/pin/hm3d/v1/val/content/00813-svBbv1Pavdk.json.gz"
target_ep_id = "32" # the id from that error

print(f"Inspecting file: {target_file}")

if not os.path.exists(target_file):
    print("Error: File does not exist!")
    exit(1)

try:
    with gzip.open(target_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    episodes = data.get("episodes", [])
    print(f"File loaded. Total episodes: {len(episodes)}")
    
    found = False
    all_ids = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        all_ids.append(eid)
        if eid == target_ep_id:
            found = True
            print(f"FOUND Episode {target_ep_id} in file!")
            print(f"  Scene ID in file: {ep['scene_id']}")
            print(f"  Start: {ep['start_position']}")
            print(f"  Goals: {ep['goals']}")
            break
            
    if not found:
        print(f"Episode {target_ep_id} NOT FOUND in file.")
        print(f"First 10 IDs in file: {all_ids[:10]}")
        if target_ep_id in all_ids:
            print("Wait, it is in data but match failed?")
            
except Exception as e:
    print(f"Error reading file: {e}")
