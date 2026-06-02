import hydra
import os
import sys
import json
import logging
import math
import time
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pver.data.builder import IndexBuilder, NegativeSampler
from pver.envs.env import PInVerifyEnv

from pver.policies.mllm_policy import MLLMPolicy
from pver.policies.random_policy import RandomPolicy
from pver.policies.clip_policy import CLIPPolicy
from pver.policies.server_client import ServerClient
from pver.eval.metrics import calculate_metrics
from pver.utils.visualizer import HTMLVisualizer

log = logging.getLogger(__name__)


def save_topdown_debug(ep_dir, ep_result, meta_data):
    """
    Save a top-down debug visualization showing goal and camera positions.
    Directions are calculated from camera-goal vectors, NOT from sector numbers.
    
    Args:
        ep_dir: Episode output directory
        ep_result: Episode result dict with transcript
        meta_data: Meta.json data containing goal_position and captures
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available, skipping top-down debug")
        return
    
    # Extract goal position (x, z from 3D coords - y is height)
    # Negate Z to convert from Habitat convention (-Z=forward) to map convention (+Y=forward/up)
    goal_pos = meta_data.get("goal_position_nominal", [0, 0, 0])
    goal_x, goal_z = goal_pos[0], -goal_pos[2]
    
    # Build capture tag -> (position, rgb_file) mapping
    # v2 meta.json may have "viewpoints" or "captures"
    captures = meta_data.get("captures") or meta_data.get("viewpoints") or []
    tag_to_info = {}
    for cap in captures:
        tag = cap.get("tag", "")
        cam_pos = cap.get("camera_position", [0, 0, 0])
        rgb_file = cap.get("rgb_filename") or cap.get("rgb", "")
        visible = cap.get("mask_meets_threshold", True)
        tag_to_info[tag] = {
            "pos": (cam_pos[0], -cam_pos[2]),
            "rgb": rgb_file,
            "basename": os.path.basename(rgb_file) if rgb_file else "",
            "visible": visible
        }
    
    def calculate_direction_from_coords(cam_x, cam_z, goal_x, goal_z):
        """Calculate direction name based on camera position relative to goal."""
        # Vector from goal to camera
        dx = cam_x - goal_x
        dz = cam_z - goal_z
        
        # Calculate angle (in degrees, 0 = +X direction, counter-clockwise positive)
        angle = math.degrees(math.atan2(dz, dx))
        
        # Normalize to 0-360
        angle = (angle + 360) % 360
        
        # Map angle to direction (assuming front = looking at goal from +X direction)
        # Adjust based on your coordinate system
        return angle
    
    # Extract visited positions from transcript using rgb_path
    transcript = ep_result.get("transcript", [])
    visited_positions = []
    
    for step_rec in transcript:
        obs = step_rec.get("observation", {})
        rgb_path = obs.get("rgb_path", "")
        action = step_rec.get("action", {})
        step_info = step_rec.get("info", {})
        nav_direction = action.get("nav_rel", "")
        current_sector_name = action.get("_current_abs_sector", "unknown")
        nav_failed = step_info.get("navigation_failed", False)
        
        # Extract filename from rgb_path
        rgb_filename = os.path.basename(rgb_path) if rgb_path else ""
        
        # Find matching capture by rgb filename
        matched_tag = None
        # Method 1: substring match (rgb field path in full rgb_path)
        for tag, info in tag_to_info.items():
            if info["rgb"] and info["rgb"] in rgb_path:
                matched_tag = tag
                break
        # Method 2: basename match
        if not matched_tag and rgb_filename:
            for tag, info in tag_to_info.items():
                if info["basename"] and info["basename"] == rgb_filename:
                    matched_tag = tag
                    break

        if matched_tag:
            info = tag_to_info[matched_tag]
            cam_x, cam_z = info["pos"]
            angle = calculate_direction_from_coords(cam_x, cam_z, goal_x, goal_z)
            visited_positions.append({
                "step": step_rec.get("step", len(visited_positions) + 1),
                "tag": matched_tag,
                "pos": (cam_x, cam_z),
                "angle": angle,
                "nav_direction": nav_direction,
                "sector_name": current_sector_name,
                "visible": info.get("visible", True),
                "nav_failed": nav_failed
            })
        else:
            # Fallback: place at goal position
            visited_positions.append({
                "step": step_rec.get("step", len(visited_positions) + 1),
                "tag": "unknown",
                "pos": (goal_x, goal_z),
                "angle": 0,
                "nav_direction": nav_direction,
                "sector_name": current_sector_name,
                "visible": None,
                "nav_failed": nav_failed
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Plot all capture positions with their tags and computed angles
    # Visible viewpoints: green circle; Trap views (not visible): red X
    has_visible = False
    has_trap = False
    for tag, info in tag_to_info.items():
        x, z = info["pos"]
        angle = calculate_direction_from_coords(x, z, goal_x, goal_z)
        visible = info.get("visible", True)
        if visible:
            ax.scatter(x, z, c='#90EE90', s=80, alpha=0.7, marker='o', edgecolors='green', linewidths=1)
            has_visible = True
        else:
            ax.scatter(x, z, c='#FFB0B0', s=80, alpha=0.7, marker='X', edgecolors='red', linewidths=1)
            has_trap = True
        vis_label = "" if visible else " [TRAP]"
        ax.annotate(f"{tag}{vis_label}\n({angle:.0f}°)", (x, z), fontsize=7, alpha=0.7, ha='center')

        # Draw line from camera to goal
        ax.plot([x, goal_x], [z, goal_z], 'k--', alpha=0.1, lw=0.5)
    
    # Legend entries for viewpoint visibility
    if has_visible:
        ax.scatter([], [], c='#90EE90', s=80, marker='o', edgecolors='green', linewidths=1, label='Visible')
    if has_trap:
        ax.scatter([], [], c='#FFB0B0', s=80, marker='X', edgecolors='red', linewidths=1, label='Trap (no mask)')
    # Nav failed legend (check if any step had nav failure)
    if any(vp.get("nav_failed") for vp in visited_positions):
        ax.scatter([], [], c='gray', s=100, marker='D', edgecolors='orange', linewidths=2, label='Nav Failed')

    # Plot goal as red star
    ax.scatter(goal_x, goal_z, c='red', s=400, marker='*', zorder=10, label='Goal')
    ax.annotate('GOAL', (goal_x, goal_z), fontsize=14, color='red', fontweight='bold',
                xytext=(10, 10), textcoords='offset points')
    
    # Plot visited positions with step numbers
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']  # Distinct colors
    for i, vp in enumerate(visited_positions):
        x, z = vp["pos"]
        step = vp["step"]
        tag = vp["tag"]
        nav = vp.get("nav_direction", "")
        sector_name = vp.get("sector_name", "")
        angle = vp.get("angle", 0)
        
        color = colors[i % len(colors)]
        step_visible = vp.get("visible", None)
        step_nav_failed = vp.get("nav_failed", False)

        # Edge color indicates visibility: green=visible, red=trap, black=unknown
        if step_nav_failed:
            edge_color = 'orange'
            marker_style = 'D'  # Diamond for failed navigation
        elif step_visible is True:
            edge_color = 'green'
            marker_style = 'o'
        elif step_visible is False:
            edge_color = 'red'
            marker_style = 'o'
        else:
            edge_color = 'black'
            marker_style = 'o'
        ax.scatter(x, z, c=color, s=300, edgecolors=edge_color, linewidths=3,
                   zorder=5, marker=marker_style)

        # Build status tag
        vis_tag = ""
        if step_nav_failed:
            vis_tag = " [NAV FAILED]"
        elif step_visible is False:
            vis_tag = " [TRAP]"
        ax.annotate(f"Step {step}\n{tag}{vis_tag}\nAngle: {angle:.0f}°",
                    (x, z), fontsize=8, ha='center',
                    xytext=(0, 25), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw arrow to next position
        if i < len(visited_positions) - 1:
            next_pos = visited_positions[i + 1]["pos"]
            ax.annotate('', xy=next_pos, xytext=(x, z),
                       arrowprops=dict(arrowstyle='->', color=color, lw=3))
            # Label with navigation direction
            mid_x = (x + next_pos[0]) / 2
            mid_z = (z + next_pos[1]) / 2
            if nav:
                ax.annotate(nav.split('(')[0].strip(), (mid_x, mid_z), fontsize=10, 
                           color=color, ha='center', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('-Z (forward)', fontsize=12)
    ax.set_title(f"Top-Down Debug View\n"
                 f"Episode: {ep_result.get('episode_id', '?')} | "
                 f"Prediction: {ep_result.get('prediction', '?')}\n"
                 f"(Arrow labels = agent's relative nav direction, agent always faces goal)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save
    debug_path = os.path.join(ep_dir, "topdown_debug.png")
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_config_path_and_name():
    """Parse config argument into absolute path and name."""
    # Check for --config or -c argument
    config_arg = None
    argv_copy = sys.argv.copy()
    for i, arg in enumerate(argv_copy):
        if arg in ['--config', '-c'] and i + 1 < len(argv_copy):
            config_arg = argv_copy[i + 1]
            sys.argv.remove(arg)
            sys.argv.remove(config_arg)
            break
        elif arg.startswith('--config='):
            config_arg = arg.split('=', 1)[1]
            sys.argv.remove(arg)
            break
        elif arg.startswith('-c='):
            config_arg = arg.split('=', 1)[1]
            sys.argv.remove(arg)
            break
    
    if config_arg:
        # Remove .yaml extension if present
        config_arg = config_arg.replace('.yaml', '')
        
        # Resolve to absolute path
        if not os.path.isabs(config_arg):
             config_arg = os.path.abspath(config_arg)
             
        config_dir = os.path.dirname(config_arg)
        config_name = os.path.basename(config_arg)
        
        # FIX: Check for and fix filenames with leading spaces
        # try:
        #     for fname in os.listdir(config_dir):
        #         if fname.startswith(" "):
        #             clean_name = fname.strip()
        #             old_path = os.path.join(config_dir, fname)
        #             new_path = os.path.join(config_dir, clean_name)
        #             print(f"DEBUG: Renaming corrupted file '{fname}' to '{clean_name}'")
        #             # Rename (this works if running with write perms)
        #             os.rename(old_path, new_path)
        # except Exception as e:
        #     print(f"DEBUG: Failed to attempt rename: {e}")
            
        return config_dir, config_name
    
    # Default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, '..', 'configs', 'agent')
    return os.path.abspath(default_dir), 'multi_view_attr'

# Get config path/name before hydra initialization
_config_path, _config_name = get_config_path_and_name()

# DEBUG: Print contents of config directory to verify file visibility
print(f"DEBUG: Config Path resolved to: {_config_path}")
print(f"DEBUG: Looking for config name: {_config_name}")
try:
    print(f"DEBUG: Directory contents: {os.listdir(_config_path)}")
except Exception as e:
    print(f"DEBUG: Error listing directory: {e}")


@hydra.main(version_base=None, config_path=_config_path, config_name=_config_name)
def main(cfg: DictConfig):
    log.info(f"Starting Evaluation with agent_type={cfg.method.get('agent_type', 'default')}")
    print(OmegaConf.to_yaml(cfg))

    # 1. Prepare Data - use index_file directly from config
    index_file = cfg.dataset.index_file
    
    # Build full path: if relative, prepend dataset.root/val/
    if not os.path.isabs(index_file):
        index_candidate = os.path.join(cfg.dataset.root, "val", index_file)
        if not os.path.exists(index_candidate):
            # Fallback to root directly
            index_candidate = os.path.join(cfg.dataset.root, index_file)
    else:
        index_candidate = index_file
    
    log.info(f"Using index file: {index_candidate}")
        
    # Load Description DB
    desc_db = {}
    if cfg.dataset.get("desc_db") and os.path.exists(cfg.dataset.desc_db):
        try:
             with open(cfg.dataset.desc_db, 'r', encoding='utf-8') as f:
                 desc_db = json.load(f)
             log.info(f"Loaded {len(desc_db)} object descriptions from {cfg.dataset.desc_db}")
        except Exception as e:
             log.error(f"Failed to load desc_db: {e}")

    # Build if absolutely missing (and not a pre-split attempt that failed)
    if not os.path.exists(index_candidate):
        log.info(f"Index {index_candidate} not found. Attempting to build standard index...")
        builder = IndexBuilder(cfg.dataset.root, cfg.dataset.capture_subdir, cfg.dataset.split)
        builder.build_index(index_candidate)
        # Assuming we need negative sampling too? 
        # For now simpler logic: Just build basic index.
        
    if not os.path.exists(index_candidate):
        log.error(f"Failed to find or build index: {index_candidate}")
        return

    # Load Dataset
    dataset = []
    with open(index_candidate, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    log.info(f"Loaded {len(dataset)} episodes from {index_candidate}")

    # Dataset Slicing for Parallel Execution
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", -1)
    
    if end_idx == -1 or end_idx > len(dataset):
        end_idx = len(dataset)
        
    if start_idx != 0 or end_idx != len(dataset):
        log.info(f"Applying dataset slice: {start_idx} to {end_idx} (Total: {end_idx - start_idx})")
        dataset = dataset[start_idx:end_idx]

    # 2. Init Env & Policy
    seed = cfg.get("seed", 42)  # Default seed = 42 for reproducibility
    env = PInVerifyEnv(dataset, max_steps=cfg.env.max_steps, sectors=cfg.env.sectors, data_root=cfg.dataset.root, view_priority=cfg.env.get("view_priority", "far"), seed=seed)
    
    client = ServerClient(
        qwen_text_url=cfg.server.qwen_text_url,
        qwen_vl_url=cfg.server.qwen_vl_url,
        gdino_url=cfg.server.gdino_url
    )
    
    # Determine method name
    method_name = cfg.method
    if hasattr(cfg.method, "name"):
         method_name = cfg.method.name
    
    log.info(f"Starting Evaluation with method={method_name}")

    if method_name == "random":
        policy = RandomPolicy(cfg)
    elif method_name == "mllm":
        policy = MLLMPolicy(cfg, client)
    elif method_name == "clip":
        policy = CLIPPolicy(cfg, client)
    elif method_name == "trained":
        from pver.policies.trained_policy import TrainedPolicy
        policy = TrainedPolicy(cfg, client)
    else:
        log.error("Unknown method")
        return

    # 3. Running Loop
    results = []
    
    # Extract dataset name from index filename for output directory organization
    # e.g. "pv_index_sectors6_500.jsonl" -> "sectors6_500"
    index_basename = os.path.splitext(os.path.basename(index_file))[0]
    if index_basename.startswith("pv_index_"):
        index_basename = index_basename.replace("pv_index_", "")
    
    # Append dataset name to output root
    # Final structure: <output.root>/<dataset_name>/<agent_name_implicitly_handled_by_config>
    # Wait, the request wants: outputs/<dataset_name>/<agent_dir>
    # In config, output.root is likely set to "outputs/<agent_name>". 
    # So we need to restructure it.
    
    # Re-evaluating path logic:
    # Current config output.root usually looks like "./outputs/single_view_direct"
    # User wants: "./outputs/sectors6_500/single_view_direct"
    
    # Respect output.root strictly from config
    output_dir = cfg.output.root

    # Support multi-GPU: append suffix if provided
    output_suffix = cfg.output.get("suffix", "")
    if output_suffix:
        output_dir = output_dir + output_suffix
        log.info(f"Multi-GPU mode: appending suffix '{output_suffix}' to output directory")

    log.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    run_start_time = time.time()

    # ---- Resume: scan for completed episodes ----
    resume = cfg.get("resume", False)
    completed_keys = set()
    resumed_results = []

    if resume and os.path.exists(output_dir):
        log.info("Resume mode: scanning for completed episodes...")
        for dirpath, _dirnames, filenames in os.walk(output_dir):
            if "episode.json" in filenames:
                ep_json_path = os.path.join(dirpath, "episode.json")
                try:
                    with open(ep_json_path, 'r', encoding='utf-8') as f:
                        ep_data = json.load(f)
                    # Path structure: output_dir / pair_type / correct|wrong / scene / ep_id
                    rel = os.path.relpath(dirpath, output_dir)
                    parts = rel.replace("\\", "/").split("/")
                    if len(parts) >= 4:
                        completed_keys.add((parts[0], parts[2], parts[3]))
                        resumed_results.append(ep_data)
                except Exception as e:
                    log.warning(f"Failed to load {ep_json_path}: {e}")
        log.info(f"Resume: found {len(completed_keys)} completed episodes, will skip them")

    def _episode_key(d_item):
        """Extract (pair_type, scene_key, episode_id) for resume matching."""
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

    skipped = 0

    for i in tqdm(range(len(dataset))):
        d_item = dataset[i]

        # Resume: skip already completed episodes
        if resume and _episode_key(d_item) in completed_keys:
            skipped += 1
            continue

        # Pre-fetch query descriptions and inject into dataset item BEFORE reset
        # Prefer query_object_id (new JSONL format) over object_id (legacy/meta.json)
        obj_id = d_item.get("query_object_id") or d_item.get("object_id")
        
        # Fetch descriptions from desc_db if not in dataset item
        if not d_item.get("query_descriptions") and not d_item.get("text_pos"):
            if obj_id and obj_id in desc_db:
                qt = desc_db[obj_id]
                if isinstance(qt, dict):
                    qt = qt.get("descriptions") or qt.get("text") or list(qt.values())
                d_item["query_descriptions"] = qt
        
        obs = env.reset(i)
        policy.reset(obs)
        
        done = False
        steps = 0
        transcript = []
        
        # Schema mapping (Legacy vs New vs Lightweight)
        # We normalize to standard keys for internal use
        # d_item is already set and potentially modified above
        ep_id = d_item.get("episode_id") or d_item.get("episode")
        # obj_id is already set above
        scene_id = d_item.get("scene_key") or d_item.get("scene")
        
        # Fallback: Extract scene from episode_path
        if not scene_id:
            ep_path = d_item.get("episode_path", "")
            # Format: pin_capture/val/00831-yr17PDCnDDW/43
            parts = ep_path.split("/")
            if len(parts) >= 3:
                scene_id = parts[-2]  # e.g., "00831-yr17PDCnDDW"
        
        
        # Extract query text for Visualization
        qt = d_item.get("query_descriptions") or d_item.get("text_pos")
        if not qt and obj_id in desc_db:
             # Structure of desc_db: {obj_id: [desc1, desc2, desc3], ...} 
             # Or {obj_id: {text: ...}}? Check file format. 
             # Standard provided format is often {obj_id: [list of strings]} or similar.
             # User said: "each object_id three sentences".
             qt = desc_db[obj_id]
             # Handle if it's a dict
             if isinstance(qt, dict):
                 qt = qt.get("descriptions") or qt.get("text") or list(qt.values())

        if not qt:
             qt = ["No description found"]*3
        
        # Windows Path Fix for Visualizer
        # If env is running on Windows but index has /root/ paths, visualizer needs valid paths or data
        # We can't change 'dataset' since Env uses it.
        # But we can patch transcript observations for visualizer OR just fix it in Visualizer?
        # Better to just not rely on absolute path if we can resolve it.
        # But Env returns absolute rgb_path.
        # Let's fix it in Visualizer by checking os.exists? Already does.
        # The issue is Env constructs path using /root/...
        


        # Get target descriptions (what's actually in the image)
        # For positive samples: same as query
        # For negative samples: use target_object_id to look up in desc_db
        pair_type = d_item.get("pair_type", "positive")
        target_object_id = d_item.get("target_object_id")
        
        if pair_type == "positive":
            target_descs = qt  # Same as query for positive samples
        else:
            # Negative sample: look up target object's descriptions
            target_descs = d_item.get("target_descriptions") or d_item.get("text_neg") or []
            if not target_descs and target_object_id and target_object_id in desc_db:
                target_descs = desc_db[target_object_id]
                if isinstance(target_descs, dict):
                    target_descs = target_descs.get("descriptions") or target_descs.get("text") or list(target_descs.values())
        
        ep_result = {
            "episode_id": ep_id,
            "scene_id": scene_id,  # For visualizer
            "object_id": obj_id,
            "target_object_id": target_object_id,  # Ground truth object in image
            "target_object_category": d_item.get("target_object_category", "unknown"),
            "query_object_category": d_item.get("query_object_category",
                                                 d_item.get("target_object_category", "unknown")),
            "prediction": "Unsure",
            "label": d_item.get("label", 1),
            "pair_type": d_item.get("pair_type", "positive"),
            "query_descriptions": qt,  # What we're searching for
            "target_descriptions": target_descs,  # What's actually in image
            "steps": 0,
            "is_correct": False,
            "transcript": []
        }
        
        while not done:
            # Save input observation BEFORE calling env.step
            input_obs = obs.copy()
            input_rgb_path = obs["rgb_path"]
            
            action = policy.act(obs)
            
            # Extract debug info from action if policy put it there
            tracker_state = action.pop("_debug_tracker", {})
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            info["tracker_state"] = tracker_state
            
            # Use INPUT observation (not output) for transcript to match action/crop
            rgb_p = input_rgb_path
            if os.name == 'nt' and rgb_p.startswith("/root"):
                 rgb_p = rgb_p.replace("./data/pv_dataset", "e:/pv_benchmark/autodl-tmp/pv_dataset")
                 rgb_p = rgb_p.replace("/", "\\")
            
            # Reconstruct Step Data - use INPUT obs for correct alignment
            step_record = {
                "step": steps,
                "observation": {"rgb_path": rgb_p},
                "action": action,
                "info": info
            }
            
            transcript.append(step_record)
            
            if done:
                ep_result["prediction"] = info["decision"]
                ep_result["steps"] = steps
        
        # Calculate correctness (Mock logic: In real pipeline, negatives are labeled 0)
        # Here we assume positive samples for now unless labeled otherwise
        label = d_item.get("label", 1)
        pred_map = {"Yes": 1, "No": 0, "Unsure": 0 if cfg.evaluation.unsure_as_negative else -1}
        
        pred_val = pred_map.get(ep_result["prediction"], -1)
        if pred_val != -1:
            ep_result["is_correct"] = (pred_val == label)
        else:
            ep_result["is_correct"] = False # Unsure penalty
            
        # Count navigation failures by type
        nav_fail_unreachable = 0
        nav_fail_trap = 0
        for rec in transcript:
            step_info = rec.get("info", {})
            if step_info.get("navigation_failed", False):
                nav_fail_unreachable += 1
            if step_info.get("trap_view", False):
                nav_fail_trap += 1
        ep_result["nav_failures"] = nav_fail_unreachable + nav_fail_trap
        ep_result["nav_fail_unreachable"] = nav_fail_unreachable
        ep_result["nav_fail_trap"] = nav_fail_trap
        ep_result["effective_views"] = steps - nav_fail_unreachable - nav_fail_trap

        # First-view prediction tracking
        step_predictions = [
            rec.get("action", {}).get("_step_prediction")
            for rec in transcript
        ]
        step_predictions = [p for p in step_predictions if p is not None]

        if step_predictions:
            ep_result["first_view_prediction"] = step_predictions[0]
        else:
            ep_result["first_view_prediction"] = ep_result["prediction"]

        ep_result["transcript"] = transcript
        results.append(ep_result)
        
        # Save per-episode log
        # Structure: output_root / pair_type / correct|wrong / scene_key / episode_id /
        scene_key = scene_id or "unknown_scene"
        pair_type = ep_result.get("pair_type", "unknown")
        correctness = "correct" if ep_result["is_correct"] else "wrong"
        
        ep_dir = os.path.join(output_dir, pair_type, correctness, scene_key, str(ep_id))
        os.makedirs(ep_dir, exist_ok=True)
        
        # Save crop images as separate files (only when save_viz is enabled)
        if cfg.output.save_viz:
            import base64
            crops_dir = os.path.join(ep_dir, "crops")
            for step_idx, step_rec in enumerate(transcript):
                crop_b64 = step_rec["action"].get("_debug_crop", "")
                if crop_b64 and crop_b64.startswith("data:image"):
                    try:
                        b64_data = crop_b64.split(",")[1]
                        img_bytes = base64.b64decode(b64_data)
                        os.makedirs(crops_dir, exist_ok=True)
                        crop_file = os.path.join(crops_dir, f"crop_step{step_idx+1}.jpg")
                        with open(crop_file, "wb") as cf:
                            cf.write(img_bytes)
                        step_rec["action"]["_debug_crop"] = f"crops/crop_step{step_idx+1}.jpg"
                    except Exception:
                        pass

        # Strip base64 crop data from episode.json to save space
        for step_rec in transcript:
            crop_val = step_rec.get("action", {}).get("_debug_crop", "")
            if crop_val and crop_val.startswith("data:image"):
                step_rec["action"].pop("_debug_crop", None)

        with open(os.path.join(ep_dir, "episode.json"), 'w') as f:
            json.dump(ep_result, f, indent=2)

        # Save sector debug log and visualizations (only when save_viz is enabled)
        if cfg.output.save_viz:
            sector_log_path = os.path.join(ep_dir, "sector_debug.log")
            with open(sector_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Episode: {ep_id} | Scene: {scene_key} | PairType: {pair_type}\n")
                f.write("=" * 80 + "\n\n")

                # Write sector map info at the beginning
                if hasattr(env, 'sector_map_debug'):
                    sm_info = env.sector_map_debug
                    f.write("[SECTOR MAP]\n")
                    f.write(f"  Total Captures: {sm_info.get('total_captures', 'N/A')}\n")
                    f.write(f"  Available Virtual Sectors: {sm_info.get('available_virtual_sectors', 'N/A')}\n")
                    f.write(f"  Starting Sector: {sm_info.get('starting_sector', 'N/A')}\n")
                    f.write(f"\n  Sector Details:\n")
                    for vsec, caps in sorted(sm_info.get('sector_details', {}).items()):
                        cap_strs = []
                        for cap_info in caps:
                            cap_strs.append(f"{cap_info['tag']}(raw={cap_info['raw_sector']},{cap_info['view_type']})")
                        f.write(f"    Virtual Sector {vsec}: {', '.join(cap_strs)}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
                for step_rec in transcript:
                    step_num = step_rec.get("step", "?")
                    action = step_rec.get("action", {})

                    f.write(f"--- Step {step_num} ---\n")

                    # Sector debug info
                    sector_debug = action.get("_sector_debug", {})
                    f.write(f"  env_returned_sector: {sector_debug.get('env_returned_sector', 'N/A')}\n")
                    f.write(f"  policy_current_abs_idx: {sector_debug.get('policy_current_abs_idx', 'N/A')}\n")
                    f.write(f"  policy_visited_abs_indices: {sector_debug.get('policy_visited_abs_indices', 'N/A')}\n")
                    f.write(f"  visited_sector_names: {sector_debug.get('visited_sector_names', 'N/A')}\n")

                    # Current state from action
                    f.write(f"  _current_abs_sector: {action.get('_current_abs_sector', 'N/A')}\n")
                    f.write(f"  _visited_abs_sectors: {action.get('_visited_abs_sectors', 'N/A')}\n")

                    conf = action.get('_detection_confidence', 'N/A')
                    if isinstance(conf, (int, float)):
                        f.write(f"  _detection_confidence: {conf:.3f}\n")
                    else:
                        f.write(f"  _detection_confidence: {conf}\n")

                    # Navigation failure info
                    step_info = step_rec.get("info", {})
                    if step_info.get("navigation_failed"):
                        f.write(f"  *** NAVIGATION FAILED: direction '{action.get('nav_rel', 'N/A')}' is UNREACHABLE ***\n")

                    # NBV debug info (if present)
                    nbv_debug = action.get("_nbv_debug", {})
                    if nbv_debug:
                        f.write(f"  [NBV Debug]\n")
                        nbv_mode = nbv_debug.get('mode', 'N/A')
                        f.write(f"    mode: {nbv_mode}\n")
                        f.write(f"    current_abs_idx (NBV): {nbv_debug.get('current_abs_idx', 'N/A')}\n")
                        f.write(f"    visited_abs_indices (NBV): {nbv_debug.get('visited_abs_indices', 'N/A')}\n")

                        # LLM NBV specific fields
                        if nbv_mode != 'farthest_point':
                            f.write(f"    visited_names_relative: {nbv_debug.get('visited_names_relative', 'N/A')}\n")
                            f.write(f"    available_directions: {nbv_debug.get('available_directions', 'N/A')}\n")
                            f.write(f"    parsed_direction: {nbv_debug.get('parsed_direction', 'N/A')}\n")
                            f.write(f"    canonical_direction: {nbv_debug.get('canonical_direction', 'N/A')}\n")
                            f.write(f"    calculated_target_sector: {nbv_debug.get('calculated_target_sector', 'N/A')}\n")
                            f.write(f"    use_dynamic_coords: {nbv_debug.get('use_dynamic_coords', 'N/A')}\n")

                        # FPS NBV specific fields
                        if nbv_mode == 'farthest_point':
                            f.write(f"    method: {nbv_debug.get('method', 'N/A')}\n")
                            f.write(f"    target_sector: {nbv_debug.get('target', 'N/A')}\n")
                            f.write(f"    relative_direction: {nbv_debug.get('relative_direction', 'N/A')}\n")

                            if 'best_min_angle_diff_deg' in nbv_debug:
                                f.write(f"    best_min_angle_diff: {nbv_debug['best_min_angle_diff_deg']:.1f}°\n")

                            if 'fallback' in nbv_debug:
                                f.write(f"    fallback: {nbv_debug['fallback']}\n")

                            # Show sector angles
                            if 'sector_angles' in nbv_debug:
                                f.write(f"    sector_angles: {nbv_debug['sector_angles']}\n")

                            # Show candidate scores summary
                            if 'candidate_scores' in nbv_debug:
                                f.write(f"    [Candidate Scores]\n")
                                for sector_id, scores in sorted(nbv_debug['candidate_scores'].items()):
                                    min_diff = scores.get('min_angle_diff_deg', 0)
                                    sec_angle = scores.get('sector_angle_deg', 'N/A')
                                    f.write(f"      Sector {sector_id} (angle={sec_angle}°): min_diff={min_diff:.1f}°\n")
                        # Write NBV prompt (truncated)
                        prompt = nbv_debug.get('prompt', '')
                        if prompt:
                            f.write(f"    [NBV Prompt Preview]:\n")
                            # Extract key parts of prompt
                            if "Already Visited" in prompt:
                                start = prompt.find("Already Visited")
                                end = prompt.find("\n", start + 50) if prompt.find("\n", start + 50) != -1 else start + 200
                                f.write(f"      ...{prompt[start:end]}...\n")
                            if "Available directions" in prompt:
                                start = prompt.find("Available directions")
                                end = prompt.find("\n", start + 50) if prompt.find("\n", start + 50) != -1 else start + 200
                                f.write(f"      ...{prompt[start:end]}...\n")

                    f.write(f"  nav_rel: {action.get('nav_rel', 'N/A')}\n")
                    f.write(f"  decision: {action.get('decision', 'N/A')}\n")

                    # Write env navigation debug info (if available)
                    if hasattr(env, 'nav_debug_log'):
                        # Find nav_debug entry for this step
                        nav_entry = None
                        for entry in env.nav_debug_log:
                            if entry.get('step') == step_num:
                                nav_entry = entry
                                break

                        if nav_entry:
                            f.write(f"\n  [ENV Navigation Debug]\n")
                            f.write(f"    Input Direction: {nav_entry.get('rel_dir_input', 'N/A')}\n")
                            f.write(f"    Normalized Direction: {nav_entry.get('rel_dir_normalized', 'N/A')}\n")
                            f.write(f"    Method: {nav_entry.get('method', 'N/A')}\n")
                            f.write(f"    Current Capture: {nav_entry.get('current_capture_tag', 'N/A')} (idx={nav_entry.get('current_obs_idx', 'N/A')})\n")

                            if 'current_angle_deg' in nav_entry:
                                f.write(f"    Current Angle: {nav_entry['current_angle_deg']:.1f}°\n")
                            if 'target_angle_deg' in nav_entry:
                                f.write(f"    Target Angle: {nav_entry['target_angle_deg']:.1f}° (offset={nav_entry.get('target_offset_deg', 0):.1f}°)\n")

                            f.write(f"    Visited Sectors: {nav_entry.get('visited_sectors', 'N/A')}\n")
                            f.write(f"    Unvisited Sectors: {nav_entry.get('unvisited_sectors', 'N/A')}\n")

                            # Candidate matching details
                            candidates = nav_entry.get('candidates', [])
                            if candidates:
                                f.write(f"    Candidates ({len(candidates)}):\n")
                                for cand in candidates:
                                    f.write(f"      Sector {cand['sector_idx']}: {cand['cap_tag']} - angle={cand['angle_deg']:.1f}°, diff={cand['angle_diff_deg']:.1f}°\n")

                            f.write(f"    Chosen Sector: {nav_entry.get('chosen_sector', 'N/A')}\n")
                            if nav_entry.get('best_angle_diff_deg') is not None:
                                f.write(f"    Best Angle Diff: {nav_entry['best_angle_diff_deg']:.1f}°\n")
                            f.write(f"    Result: {nav_entry.get('result', 'N/A')}\n")
                            f.write(f"    Result Capture Index: {nav_entry.get('result_cap_idx', 'N/A')}\n")
                            if 'fallback' in nav_entry:
                                f.write(f"    Fallback: {nav_entry['fallback']}\n")

                    f.write("\n")

            # vis.html
            viz_path = os.path.join(ep_dir, "vis.html")
            visualizer = HTMLVisualizer() # Re-init or reuse? Lightweight enough to re-init or make global
            visualizer.save_episode(ep_result, viz_path)
            
            # 5.1 Top-down debug visualization
            # Construct meta.json path from dataset info
            meta_rel_path = d_item.get("meta_path")
            if meta_rel_path and cfg.dataset.root:
                full_meta_path = os.path.join(cfg.dataset.root, meta_rel_path)
                # Windows path fix
                if os.name == 'nt':
                    full_meta_path = full_meta_path.replace("/", "\\")
                if os.path.exists(full_meta_path):
                    try:
                        with open(full_meta_path, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        save_topdown_debug(ep_dir, ep_result, meta_data)
                    except Exception as e:
                        print(f"[WARN] Failed to create top-down debug: {e}")

    # 4. Save results and summarize
    if resume and skipped > 0:
        log.info(f"Resume complete: skipped {skipped} existing, ran {len(results)} new episodes")

    # Combine resumed + new results
    if resume:
        all_results = resumed_results + results
    else:
        # Original behavior: load existing results.json for append mode
        existing_results = []
        results_file = os.path.join(output_dir, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                log.info(f"Loaded {len(existing_results)} existing results from {results_file}")
            except Exception as e:
                log.warning(f"Failed to load existing results: {e}")
                existing_results = []
        all_results = existing_results + results

    results_file = os.path.join(output_dir, "results.json")

    # Save combined results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(all_results)} total episode results to {results_file} ({len(results)} new)")

    # Calculate metrics on ALL results (not just new ones)
    metrics = calculate_metrics(all_results)

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

    # Record wall time
    wall_time_s = time.time() - run_start_time
    metrics["wall_time_s"] = round(wall_time_s, 1)
    print(f"\nWall time: {wall_time_s:.1f}s ({wall_time_s/60:.1f}min)")

    # Inject config info into metrics for self-contained results
    metrics["config_info"] = {
        "agent_type": cfg.method.get("agent_type", "unknown"),
        "nbv_type": cfg.method.get("nbv", {}).get("type", "none"),
        "fusion_type": cfg.method.get("fusion", {}).get("type", "none"),
        "bbox_mode": cfg.method.get("bbox_mode", "unknown"),
        "max_steps": cfg.env.get("max_steps", 1),
        "query_mode": cfg.method.get("query_mode", "unknown"),
    }

    # Save metrics (summary)
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    # Prevent Hydra from creating outputs/<date>/<time>/ dirs, .hydra/ subdir, and job log file
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.job_logging.root.handlers=[]")
    main()
