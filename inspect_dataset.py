import gzip
import json
import os

input_file = "./data/datasets/pin/hm3d/v1/val/content/00800-TEEsavR23oF.json.gz"
output_file = "episode_sample.json"

try:
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
        sample = data["episodes"][0]
        
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(sample, out, indent=2)
            
    print(f"Sample episode saved to {output_file}")
    
except Exception as e:
    print(f"Error: {e}")
