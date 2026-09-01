import os
import glob
import json
import gzip
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================================================
# Point this at the dataset's content directory
# =========================================================
# e.g. <dataset root>/datasets/pin/hm3d/v1/val/content
DATASET_CONTENT_DIR = os.environ.get(
    "PIN_CONTENT_DIR", "./data/datasets/pin/hm3d/v1/val/content")

def analyze_dataset_distribution(content_dir):
    """
    Count episodes per scene straight from the dataset json.gz files.
    """
    print(f"Scanning dataset files in: {content_dir}")
    
    # find every json.gz
    pattern = os.path.join(content_dir, "*.json.gz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No *.json.gz files found!")
        return

    # count episodes per scene
    scene_counts = defaultdict(int)
    total_episodes = 0
    
    print(f"Found {len(files)} scene files. Reading...")
    
    for fpath in files:
        scene_name = os.path.basename(fpath).replace(".json.gz", "")
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
                
            # Habitat layout: {"episodes": [...]}
            episodes = data.get("episodes", [])
            count = len(episodes)
            
            scene_counts[scene_name] = count
            total_episodes += count
            
            # print(f"  {scene_name}: {count}")
            
        except Exception as e:
            print(f"Error reading {scene_name}: {e}")
            continue

    if total_episodes == 0:
        print("No episodes found.")
        return

    # sort
    sorted_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)
    
    num_scenes = len(sorted_scenes)
    avg_eps = total_episodes / num_scenes if num_scenes > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 Dataset Distribution Summary")
    print(f"{'='*60}")
    print(f"Total Scenes:    {num_scenes}")
    print(f"Total Episodes:  {total_episodes}")
    print(f"Avg Eps/Scene:   {avg_eps:.2f}")
    if sorted_scenes:
        print(f"Max Eps in Scene: {sorted_scenes[0][1]} ({sorted_scenes[0][0]})")
        print(f"Min Eps in Scene: {sorted_scenes[-1][1]} ({sorted_scenes[-1][0]})")
    print(f"{'='*60}")
    
    print("\nTop 10 Most Populated Scenes:")
    for name, count in sorted_scenes[:10]:
        print(f"  {name:25s}: {count}")

    print("\nTop 10 Least Populated Scenes:")
    for name, count in sorted_scenes[-10:]:
        print(f"  {name:25s}: {count}")

    # =========================================================
    # plot the distribution
    # =========================================================
    plt.figure(figsize=(15, 6))
    
    # bar chart
    counts = [count for _, count in sorted_scenes]
    
    plt.bar(range(len(counts)), counts, color='#3498db', edgecolor='black', alpha=0.7)
    
    plt.axhline(y=avg_eps, color='r', linestyle='--', label=f'Average ({avg_eps:.1f})')
    plt.xlabel('Scene Index (Sorted by Count)')
    plt.ylabel('Number of Episodes')
    plt.title(f'Episode Distribution per Scene (Total: {len(files)} Scenes)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# entry point
analyze_dataset_distribution(DATASET_CONTENT_DIR)
