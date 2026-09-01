#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object penetration detection for the PIN dataset.

Checks every episode in a split for objects intersecting scene geometry, and can photograph each one.
Method: Habitat's contact-point API for collision detection.

Usage:
    python detect_penetration.py --config configs/models/pin/pin_hm3d_v1.yaml --split val --max-episodes 10

Outputs:
    - penetration_report_{split}.json: the report
    - snapshots/{scene}/{episode}_*.png: photographs taken near the object
"""

import os
import sys
import json
import argparse
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from tqdm import tqdm
import imageio
import quaternion as nq

import habitat
from habitat_sim.gfx import NO_LIGHT_KEY
from habitat_sim.utils.common import quat_from_two_vectors


def normalize_np_quat(q: nq.quaternion) -> nq.quaternion:
    """Normalize a quaternion; habitat rejects non-unit ones."""
    n = float(np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z))
    if n <= 1e-12:
        return nq.quaternion(1.0, 0.0, 0.0, 0.0)
    return nq.quaternion(q.w/n, q.x/n, q.y/n, q.z/n)


def capture_object_photo(
    sim,
    object_pos: np.ndarray,
    distance: float = 1.5,
    sensor_offset_y: float = 1.31,
) -> Optional[np.ndarray]:
    """
    Photograph the object from nearby, following the capture logic of the collection pipeline.
    
    Sets the pose with sim.set_agent_state(), then reads sim.get_sensor_observations().
    Yaw and pitch are derived from the target position so it lands in frame centre.
    
    Args:
        sim: the Habitat simulator
        object_pos: object position [x, y, z]
        distance: shooting distance
        sensor_offset_y: camera height above the agent's feet
    
    Returns:
        An RGB array, or None
    """
    import magnum as mn
    
    def compute_look_at_rotation(cam_pos: np.ndarray, target_pos: np.ndarray) -> nq.quaternion:
        """
        Full rotation looking from cam_pos toward target_pos, yaw and pitch.
        
        Habitat convention: Y is up, the camera faces -Z by default.
        """
        # direction vector
        direction = target_pos - cam_pos
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return nq.quaternion(1.0, 0.0, 0.0, 0.0)
        
        direction = direction / dist
        
        # horizontal distance and height difference
        horizontal_dist = float(np.sqrt(direction[0]**2 + direction[2]**2))
        height_diff = float(direction[1])
        
        # yaw, about the Y axis
        # horizontal direction
        dir_horizontal = np.array([direction[0], 0.0, direction[2]], dtype=np.float32)
        h_norm = float(np.linalg.norm(dir_horizontal))
        if h_norm > 1e-6:
            dir_horizontal = dir_horizontal / h_norm
        else:
            dir_horizontal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        yaw_quat = quat_from_two_vectors(forward, dir_horizontal)
        yaw_quat = normalize_np_quat(yaw_quat)
        
        # pitch, about the X axis
        # height_diff > 0: target is above, look up (negative pitch)
        # height_diff < 0: target is below, look down (positive pitch)
        if horizontal_dist > 1e-6:
            pitch_angle = float(np.arctan2(height_diff, horizontal_dist))
        else:
            pitch_angle = 0.0
        
        # pitch quaternion about the local X axis
        # Habitat sets sensor pitch through the sensor orientation;
        # here it is folded into the agent rotation instead
        pitch_quat = nq.quaternion(
            float(np.cos(pitch_angle / 2)),
            float(np.sin(pitch_angle / 2)),
            0.0,
            0.0
        )
        
        # compose: yaw first, then pitch
        # final_quat = yaw_quat * pitch_quat
        final_quat = yaw_quat * pitch_quat
        return normalize_np_quat(final_quat)
    
    try:
        # make sure object_pos is a proper array
        obj_pos = np.array([float(object_pos[0]), float(object_pos[1]), float(object_pos[2])], dtype=np.float32)
        
        # try several directions and distances around the object for a navigable spot
        angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, 
                  np.pi, 7*np.pi/6, 5*np.pi/4, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 7*np.pi/4, 11*np.pi/6]
        distances = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # candidate distances
        
        for dist in distances:
            for angle in angles:
                # camera position on a circle around the object
                dx = dist * np.cos(angle)
                dz = dist * np.sin(angle)
                
                # camera position, offset horizontally
                camera_pos_attempt = np.array([
                    obj_pos[0] + dx,
                    obj_pos[1],  # object height for now, snapped to the navmesh below
                    obj_pos[2] + dz,
                ], dtype=np.float32)
                
                # is the foot position navigable
                if not sim.pathfinder.is_navigable(camera_pos_attempt):
                    continue
                
                # snap to the navmesh for the foot position
                snapped_pos = sim.pathfinder.snap_point(camera_pos_attempt)
                
                # agent foot position
                agent_base_pos = np.array([
                    float(snapped_pos[0]),
                    float(snapped_pos[1]),
                    float(snapped_pos[2]),
                ], dtype=np.float32)
                
                # camera sits at foot + sensor_offset_y
                camera_pos = np.array([
                    agent_base_pos[0],
                    agent_base_pos[1] + sensor_offset_y,
                    agent_base_pos[2],
                ], dtype=np.float32)
                
                # full look-at rotation
                look_at_quat = compute_look_at_rotation(camera_pos, obj_pos)
                
                # set the agent state, using magnum types
                sim.set_agent_state(
                    position=mn.Vector3(float(agent_base_pos[0]), float(agent_base_pos[1]), float(agent_base_pos[2])),
                    rotation=look_at_quat,
                )
                
                # read the sensors
                obs = sim.get_sensor_observations()
                rgb_frame = obs.get("rgb", None)
                
                if rgb_frame is not None:
                    # to uint8
                    if rgb_frame.dtype != np.uint8:
                        if rgb_frame.dtype in (np.float16, np.float32, np.float64):
                            rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
                        else:
                            rgb_frame = rgb_frame.astype(np.uint8)
                    return rgb_frame
        
        # no navigable spot in any direction
        return None
            
    except Exception as e:
        print(f"  Warning: Failed to capture photo: {e}")
        return None


def detect_penetration_for_episode(
    sim,
    rigid_obj_mgr,
    obj_templates_mgr,
    episode,
    save_photo: bool = False,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Test one episode's target object for penetration, optionally photographing it.
    
    Uses the contact test to find collisions with scene geometry.
    """
    
    result = {
        "scene_id": episode.scene_id,
        "episode_id": str(episode.episode_id),
        "object_id": None,
        "object_category": None,
        "goal_position": None,
        "is_penetrating": False,
        "snap_down_success": False,
        "snapped_position": None,
        "position_delta": None,
        "error": None,
        "snapshot_path": None,
    }
    
    try:
        # target object information
        goal = episode.goals[0]
        object_id = str(goal.object_id)
        object_category = goal.object_category
        goal_position = list(goal.position)
        
        result["object_id"] = object_id
        result["object_category"] = object_category
        result["goal_position"] = [float(x) for x in goal_position]
        
        # clear any object left from the previous episode
        rigid_obj_mgr.remove_all_objects()
        
        # load the object template
        obj_handle_list = obj_templates_mgr.get_template_handles(object_id)
        if not obj_handle_list:
            result["error"] = f"Object template not found: {object_id}"
            return result
        
        obj_handle = obj_handle_list[0]
        
        # add the object to the scene
        obj = rigid_obj_mgr.add_object_by_template_handle(
            obj_handle, light_setup_key=NO_LIGHT_KEY
        )
        
        if obj is None:
            result["error"] = f"Failed to add object: {object_id}"
            return result
        
        # place it
        obj.translation = np.array(goal_position)
        obj.semantic_id = 100000
        
        # photograph before the test, while it is still in its original position
        if save_photo and snapshot_dir:
            scene_name = os.path.basename(os.path.dirname(episode.scene_id))
            scene_snap_dir = os.path.join(snapshot_dir, scene_name)
            os.makedirs(scene_snap_dir, exist_ok=True)
            
            photo = capture_object_photo(sim, np.array(goal_position))
            if photo is not None:
                snap_path = os.path.join(
                    scene_snap_dir,
                    f"ep{episode.episode_id}_{object_category}.png"
                )
                imageio.imwrite(snap_path, photo)
                result["snapshot_path"] = snap_path
        
        # read the physics contact points
        # contact_distance < -0.01 means more than 1 cm of penetration; negative is inside
        # a normal resting contact is >= 0
        
        # run discrete collision detection
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        
        # keep the contacts involving the target
        max_penetration_depth = 0.0
        penetrating_contacts = []
        
        for cp in contact_points:
            # does this contact involve the target
            if cp.object_id_a == obj.object_id or cp.object_id_b == obj.object_id:
                contact_dist = float(cp.contact_distance)
                
                # negative distance means penetration
                # 1 cm threshold, so small physics noise is ignored
                if contact_dist < -0.01:
                    penetration_depth = -contact_dist
                    if penetration_depth > max_penetration_depth:
                        max_penetration_depth = penetration_depth
                    penetrating_contacts.append({
                        "object_id_a": cp.object_id_a,
                        "object_id_b": cp.object_id_b,
                        "contact_distance": contact_dist,
                        "penetration_depth_cm": penetration_depth * 100,
                    })
        
        result["max_penetration_depth"] = float(max_penetration_depth)
        result["num_penetrating_contacts"] = len(penetrating_contacts)
        
        if len(penetrating_contacts) > 0:
            # penetrating
            result["is_penetrating"] = True
            result["snap_down_success"] = False
            result["error"] = f"Penetration detected: max depth {max_penetration_depth*100:.2f}cm, {len(penetrating_contacts)} contacts"
        else:
            # resting normally
            result["is_penetrating"] = False
            result["snap_down_success"] = True
        
        # clean up
        rigid_obj_mgr.remove_all_objects()
        
    except Exception as e:
        result["error"] = str(e)
        result["is_penetrating"] = True
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Object penetration detection for the PIN dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/pin/pin_hm3d_v1.yaml",
        help="Path to the config yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./penetration_results",
        help="Output directory",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap on episodes checked, for a quick pass",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split",
    )
    parser.add_argument(
        "--save-photos",
        action="store_true",
        default=False,
        help="Save photographs of each object",
    )
    
    args, unknown = parser.parse_known_args()
    args.opts = [o for o in unknown if "=" in o]
    return args


def main():
    args = parse_args()
    
    # output directory
    os.makedirs(args.output_dir, exist_ok=True)
    snapshot_dir = os.path.join(args.output_dir, "snapshots") if args.save_photos else None
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
    
    # load the config
    print(f"Loading config from: {args.config}")
    opts = args.opts + [f"habitat.dataset.split={args.split}"]
    config = habitat.get_config(args.config, opts)
    
    print(f"Dataset split: {config.habitat.dataset.split}")
    print(f"Data path: {config.habitat.dataset.data_path}")
    
    # load the dataset
    print("Loading dataset...")
    from habitat.datasets import make_dataset
    
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    
    print(f"Loaded {len(dataset.episodes)} episodes")
    
    # group by scene
    episodes_by_scene = defaultdict(list)
    for ep in dataset.episodes:
        episodes_by_scene[ep.scene_id].append(ep)
    
    print(f"Found {len(episodes_by_scene)} unique scenes")
    
    # counters
    total_episodes = 0
    penetrating_episodes = []
    valid_episodes = []
    errors = []
    category_stats = defaultdict(lambda: {"total": 0, "penetrating": 0})
    
    # create the simulator
    print("Creating simulator...")
    from habitat.sims import make_sim
    from habitat.config import read_write
    
    # apply the episode cap
    max_eps = args.max_episodes or len(dataset.episodes)
    episodes_to_check = dataset.episodes[:max_eps]
    
    print(f"Will check {len(episodes_to_check)} episodes")
    print(f"Save photos: {args.save_photos}")
    
    # process scene by scene
    pbar = tqdm(total=len(episodes_to_check), desc="Detecting penetration")
    
    current_scene = None
    sim = None
    
    for ep in episodes_to_check:
        # reload when the scene changes
        if current_scene != ep.scene_id:
            if sim is not None:
                sim.close()
            
            with read_write(config.habitat.simulator):
                config.habitat.simulator.scene_dataset = ep.scene_dataset_config
                config.habitat.simulator.scene = ep.scene_id
            
            sim = make_sim(
                id_sim=config.habitat.simulator.type,
                config=config.habitat.simulator,
            )
            current_scene = ep.scene_id
            
            # load the object template
            obj_templates_mgr = sim.get_object_template_manager()
            obj_path = config.habitat.task.objects_path.replace(
                "{split}", config.habitat.dataset.split
            )
            if os.path.isdir(obj_path):
                obj_templates_mgr.load_configs(obj_path, True)
        
        # object managers
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr = sim.get_object_template_manager()
        
        # run the penetration test
        result = detect_penetration_for_episode(
            sim=sim,
            rigid_obj_mgr=rigid_obj_mgr,
            obj_templates_mgr=obj_templates_mgr,
            episode=ep,
            save_photo=args.save_photos,
            snapshot_dir=snapshot_dir,
        )
        
        total_episodes += 1
        category = result.get("object_category", "unknown")
        category_stats[category]["total"] += 1
        
        if result["is_penetrating"]:
            penetrating_episodes.append(result)
            category_stats[category]["penetrating"] += 1
        else:
            valid_episodes.append(result)
        
        if result.get("error"):
            errors.append({
                "scene_id": result["scene_id"],
                "episode_id": result["episode_id"],
                "error": result["error"],
            })
        
        pbar.update(1)
        pbar.set_postfix({
            "penetrating": len(penetrating_episodes),
            "rate": f"{len(penetrating_episodes)/total_episodes*100:.1f}%"
        })
    
    pbar.close()
    
    if sim is not None:
        sim.close()
    
    # build the report
    print("\n" + "=" * 60)
    print("📊 PENETRATION DETECTION REPORT")
    print("=" * 60)
    print(f"Total episodes checked: {total_episodes}")
    print(f"Penetrating episodes: {len(penetrating_episodes)} ({len(penetrating_episodes)/total_episodes*100:.2f}%)")
    print(f"Valid episodes: {len(valid_episodes)} ({len(valid_episodes)/total_episodes*100:.2f}%)")
    print(f"Errors encountered: {len(errors)}")
    
    print("\n📦 Category Breakdown:")
    for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["penetrating"]):
        pen_rate = stats["penetrating"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['penetrating']}/{stats['total']} ({pen_rate:.1f}%)")
    
    # write the report
    report = {
        "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "total_episodes": total_episodes,
        "penetrating_count": len(penetrating_episodes),
        "penetrating_rate": len(penetrating_episodes) / total_episodes if total_episodes > 0 else 0,
        "valid_count": len(valid_episodes),
        "category_stats": dict(category_stats),
        "penetrating_episodes": penetrating_episodes,
        "errors": errors,
    }
    
    output_path = os.path.join(args.output_dir, f"penetration_report_{args.split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Report saved to: {output_path}")
    
    # write the list of penetrating episodes
    penetrating_list_path = os.path.join(args.output_dir, f"penetrating_episodes_{args.split}.json")
    with open(penetrating_list_path, "w", encoding="utf-8") as f:
        json.dump({
            "count": len(penetrating_episodes),
            "episodes": [
                {
                    "scene_id": ep["scene_id"],
                    "episode_id": ep["episode_id"],
                    "object_id": ep["object_id"],
                    "object_category": ep["object_category"],
                    "snapshot_path": ep.get("snapshot_path"),
                    "error": ep.get("error"),
                }
                for ep in penetrating_episodes
            ]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📁 Penetrating episodes list saved to: {penetrating_list_path}")
    
    if snapshot_dir:
        print(f"📷 Snapshots saved to: {snapshot_dir}")
    
    return report


if __name__ == "__main__":
    main()
