#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import jsonlines
import habitat
from tqdm import tqdm
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any

import imageio
import numpy as np
import math
import collections
import quaternion as nq
import magnum as mn

from habitat_sim.utils.common import quat_from_two_vectors

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# Run from inside a PIN checkout; this makes its modules importable
# regardless of where the repository lives.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("cwd:", os.getcwd())
print("Python search path:")
for p in sys.path:
    print(p)

from utils.distributed_env import DistributedEnv
from habitat.envs.habitat_pin_env import HabitatPINEnv
from utils.new_top_down_map import *  # unused here, kept for downstream extensions
from utils.oracle_navigators import ShortestPathFollowerAgentPIN
from utils.wandb_logger import PINDistributedWandbLogger


# ============================================================
# Constants
# ============================================================

TARGET_SEMANTIC_ID = 100000
TRUNC_MARGIN_THRESH_PX = 2
MIN_AREA_PIXELS = 100
GOAL_UV_TOL_PX = 2
FALLBACK_MIN_MASK_AREA_FRAC = 0.01

# Gradient-based groove filling
MAX_GAP_PX = 25             # search radius in pixels for an opposing edge
OPPOSITE_DOT_THRESH = -0.85 # normals below this dot product count as opposing
LINE_SAMPLE_STEPS = 40      # samples taken along the line to check for a background channel between edges

# video frame rate, for converting frames to seconds
VIDEO_FPS = 10

# ============================================================
# Cross-floor detection, using the OVON threshold method
# ============================================================
# HM3D's semantic annotations are unavailable at Habitat runtime
# (semantic_annotations() returns an empty object), so OVON's height threshold is used instead.

SINGLE_FLOOR_THRESHOLD = 0.25  # height spread in metres above which the path counts as crossing floors


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def check_cross_floor_by_path(sim, start_position, goal_position, threshold=SINGLE_FLOOR_THRESHOLD):
    """
    OVON style: decide from the height spread along the shortest path.
    
    This catches a path that goes up and back down to the same height.
    
    Args:
        sim: the Habitat simulator
        start_position: start [x, y, z]
        goal_position: goal [x, y, z]
        threshold: height spread in metres
    
    Returns:
        dict:
            - is_cross_floor_path: bool
            - path_height_range: float, highest minus lowest
    """
    import habitat_sim
    
    result = {
        "is_cross_floor_path": False,
        "path_height_range": 0.0,
        "path_found": False,
    }
    
    try:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(start_position, dtype=np.float32)
        path.requested_end = np.array(goal_position, dtype=np.float32)
        found_path = sim.pathfinder.find_path(path)
        
        result["path_found"] = found_path
        
        if found_path and len(path.points) > 0:
            heights = [float(p[1]) for p in path.points]
            h_delta = max(heights) - min(heights)
            result["path_height_range"] = round(h_delta, 4)
            
            if h_delta > threshold:
                result["is_cross_floor_path"] = True
    except Exception:
        pass
    
    return result


def check_cross_floor_by_trajectory(trajectory_heights, threshold=SINGLE_FLOOR_THRESHOLD):
    """
    Simplified: decide from the height spread of the walked trajectory.
    
    Args:
        trajectory_heights: the agent's Y coordinate at each step
        threshold: height spread in metres
    
    Returns:
        A dict describing the trajectory heights
    """
    result = {
        "is_cross_floor_trajectory": False,
        "trajectory_height_range": 0.0,
        "start_height": None,
        "end_height": None,
        "min_height": None,
        "max_height": None,
    }
    
    if len(trajectory_heights) < 1:
        return result
    
    heights = [float(h) for h in trajectory_heights]
    result["start_height"] = heights[0]
    result["end_height"] = heights[-1]
    result["min_height"] = min(heights)
    result["max_height"] = max(heights)
    
    h_delta = result["max_height"] - result["min_height"]
    result["trajectory_height_range"] = round(h_delta, 4)
    
    if h_delta > threshold:
        result["is_cross_floor_trajectory"] = True
    
    return result


def analyze_cross_floor(sim, start_position, goal_position, trajectory_heights, threshold=SINGLE_FLOOR_THRESHOLD):
    """
    Combined cross-floor analysis: OVON path detection plus trajectory height.
    
    Thresholds are used rather than semantic annotations because HM3D's floor
    data is unavailable at Habitat runtime (semantic_annotations() returns empty).
    
    Args:
        sim: the Habitat simulator
        start_position: episode start
        goal_position: goal
        trajectory_heights: walked heights
        threshold: height spread in metres, default 0.25
    
    Returns:
        A dict with the full cross-floor analysis
    """
    # 1. OVON style: inspect the shortest path
    path_result = check_cross_floor_by_path(sim, start_position, goal_position, threshold)
    
    # 2. inspect the actual trajectory
    traj_result = check_cross_floor_by_trajectory(trajectory_heights, threshold)
    
    # 3. either signal is enough to call it cross-floor
    is_cross_floor = path_result["is_cross_floor_path"] or traj_result["is_cross_floor_trajectory"]
    
    # assemble the explanation
    reasons = []
    if path_result["is_cross_floor_path"]:
        reasons.append(f"path_height_range={path_result['path_height_range']:.3f}m")
    if traj_result["is_cross_floor_trajectory"]:
        reasons.append(f"trajectory_height_range={traj_result['trajectory_height_range']:.3f}m")
    
    return {
        "is_cross_floor": is_cross_floor,
        "cross_floor_reason": "; ".join(reasons) if reasons else None,
        "threshold_used": threshold,
        # path detection result
        "is_cross_floor_path": path_result["is_cross_floor_path"],
        "path_height_range": path_result["path_height_range"],
        "path_found": path_result["path_found"],
        # trajectory detection result
        "is_cross_floor_trajectory": traj_result["is_cross_floor_trajectory"],
        "trajectory_height_range": traj_result["trajectory_height_range"],
        "height_range": traj_result["trajectory_height_range"],  # kept for backward compatibility
        # heights
        "start_height": traj_result["start_height"],
        "end_height": traj_result["end_height"],
        "min_height": traj_result["min_height"],
        "max_height": traj_result["max_height"],
        # kept for backward compatibility
        "num_floor_clusters": 2 if is_cross_floor else 1,
        "floor_heights": [],
        "floor_levels": [],
        "detection_method": "ovon_threshold",
        "stair_segments": [],
        "total_stair_steps": 0,
        "num_stair_segments": 0,
        "max_single_climb": traj_result["trajectory_height_range"] if is_cross_floor else 0.0,
        "max_single_descent": traj_result["trajectory_height_range"] if is_cross_floor else 0.0,
    }


def normalize_np_quat(q: nq.quaternion) -> nq.quaternion:
    n = float(np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z))
    if n <= 1e-12:
        return nq.quaternion(1.0, 0.0, 0.0, 0.0)
    return nq.quaternion(q.w/n, q.x/n, q.y/n, q.z/n)


def npquat_to_magnum(q_np: nq.quaternion) -> mn.Quaternion:
    return mn.Quaternion(
        mn.Vector3(float(q_np.x), float(q_np.y), float(q_np.z)),
        float(q_np.w),
    )


def camera_axes_from_state(cam_state):
    q = mn.Quaternion(
        mn.Vector3(
            float(cam_state.rotation.x),
            float(cam_state.rotation.y),
            float(cam_state.rotation.z),
        ),
        float(cam_state.rotation.w),
    )
    fwd = np.array(q.transform_vector(mn.Vector3(0, 0, -1)), dtype=np.float32)
    right = np.array(q.transform_vector(mn.Vector3(1, 0, 0)), dtype=np.float32)
    up = np.array(q.transform_vector(mn.Vector3(0, 1, 0)), dtype=np.float32)
    pos = np.array(cam_state.position, dtype=np.float32)
    return pos, fwd, right, up


def frustum_check(
    cam_pos,
    fwd,
    right,
    up,
    tan_h,
    tan_v,
    goal_pos,
    near=0.05,
    far=100.0,
    eps=1e-6,
) -> bool:
    v = np.asarray(goal_pos, np.float32) - np.asarray(cam_pos, np.float32)
    f = float(np.dot(fwd, v))
    if not (f > near and f < far):
        return False
    rx = float(np.dot(right, v))
    ry = float(np.dot(up, v))
    return (abs(rx) <= f * tan_h + eps) and (abs(ry) <= f * tan_v + eps)


def get_fov_tans_from_cfg(cfg) -> Tuple[float, float]:
    """
    Read HFOV and VFOV from the config and return tan(hfov/2), tan(vfov/2).
    """
    try:
        rgb_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    except Exception:
        rgb_cfg = cfg.simulator.agents.main_agent.sim_sensors.rgb_sensor
    hfov = float(rgb_cfg.hfov)
    if hasattr(rgb_cfg, "vfov") and float(getattr(rgb_cfg, "vfov", 0.0)) > 0:
        vfov = float(rgb_cfg.vfov)
    else:
        W, H = int(rgb_cfg.width), int(rgb_cfg.height)
        vfov = math.degrees(
            2 * math.atan(
                math.tan(math.radians(hfov) / 2.0) * (H / W)
            )
        )
    return math.tan(math.radians(hfov) * 0.5), math.tan(math.radians(vfov) * 0.5)


def build_extrinsics(cam_state):
    cam_q_np = normalize_np_quat(cam_state.rotation)
    cam_q_mn = npquat_to_magnum(cam_q_np)
    cam_p_mn = mn.Vector3(
        float(cam_state.position[0]),
        float(cam_state.position[1]),
        float(cam_state.position[2]),
    )
    R3 = cam_q_mn.to_matrix()
    T_c2w = mn.Matrix4.from_(R3, cam_p_mn)
    T_w2c = T_c2w.inverted()
    return cam_q_mn, T_c2w, T_w2c


def approximate_intrinsics_from_fov(W: int, H: int, tan_h: float, tan_v: float) -> np.ndarray:
    fx = 0.5 * W / tan_h
    fy = 0.5 * H / tan_v
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def project_world_point(pt_world: np.ndarray, T_w2c: mn.Matrix4, K: np.ndarray) -> Tuple[float, float, float]:
    pc = T_w2c.transform_point(mn.Vector3(float(pt_world[0]), float(pt_world[1]), float(pt_world[2])))
    z = float(pc.z)
    z_safe = z if abs(z) > 1e-6 else 1e-6
    u = float(K[0, 0] * (pc.x / z_safe) + K[0, 2])
    v = float(K[1, 1] * (pc.y / z_safe) + K[1, 2])
    return u, v, z


def fill_holes(mask_bool: np.ndarray) -> np.ndarray:
    """
    Fill fully enclosed holes. Only inside the mask; never dilates outward.
    """
    H, W = mask_bool.shape
    visited = np.zeros((H, W), dtype=bool)
    q = collections.deque()

    # flood from the background at the border
    for x in range(W):
        if not mask_bool[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if not mask_bool[H - 1, x] and not visited[H - 1, x]:
            visited[H - 1, x] = True
            q.append((H - 1, x))
    for y in range(H):
        if not mask_bool[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if not mask_bool[y, W - 1] and not visited[y, W - 1]:
            visited[y, W - 1] = True
            q.append((y, W - 1))

    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny, nx] and not mask_bool[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    # unreached black pixels are the holes
    filled = mask_bool.copy()
    holes = (~visited) & (~mask_bool)
    filled[holes] = True
    return filled


def count_connected_components(mask_bool: np.ndarray) -> int:
    H, W = mask_bool.shape
    visited = np.zeros((H, W), dtype=bool)
    cc = 0
    for y in range(H):
        for x in range(W):
            if mask_bool[y, x] and not visited[y, x]:
                cc += 1
                dq = collections.deque()
                dq.append((y, x))
                visited[y, x] = True
                while dq:
                    cy, cx = dq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if mask_bool[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                dq.append((ny, nx))
    return cc


def compute_gradient(mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = mask_bool.astype(np.float32)
    gx = (np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)) * 0.5
    gy = (np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)) * 0.5
    return gx, gy


def get_boundary_pixels(mask_bool: np.ndarray):
    H, W = mask_bool.shape
    boundary = []
    for y in range(H):
        for x in range(W):
            if not mask_bool[y, x]:
                continue
            if (y > 0 and not mask_bool[y - 1, x]) or \
               (y < H - 1 and not mask_bool[y + 1, x]) or \
               (x > 0 and not mask_bool[y, x - 1]) or \
               (x < W - 1 and not mask_bool[y, x + 1]):
                boundary.append((y, x))
    return boundary


def line_clear(mask_bool: np.ndarray, p0, p1) -> bool:
    y0, x0 = p0
    y1, x1 = p1
    for t in np.linspace(0.0, 1.0, LINE_SAMPLE_STEPS):
        y = int(round(y0 + (y1 - y0) * t))
        x = int(round(x0 + (x1 - x0) * t))
        if y < 0 or y >= mask_bool.shape[0] or x < 0 or x >= mask_bool.shape[1]:
            return False
        if 0.0 < t < 1.0 and mask_bool[y, x]:
            return False
    return True


def fill_concave_gaps_by_gradient(mask_bool: np.ndarray) -> np.ndarray:
    """
    Core of the third gate:
    - estimate boundary normals from the gradient
    - opposing normals with a background channel between them get bridged
    """
    H, W = mask_bool.shape
    gx, gy = compute_gradient(mask_bool)
    boundary = get_boundary_pixels(mask_bool)

    if len(boundary) == 0:
        return mask_bool

    mask_out = mask_bool.copy()

    for (y, x) in boundary:
        vx = gx[y, x]
        vy = gy[y, x]
        vnorm = math.hypot(vx, vy)
        if vnorm < 1e-3:
            continue
        nx = vx / vnorm
        ny = vy / vnorm

        for (y2, x2) in boundary:
            if y2 == y and x2 == x:
                continue
            dy = y2 - y
            dx = x2 - x
            dist = math.hypot(dx, dy)
            if dist < 1.0 or dist > MAX_GAP_PX:
                continue

            vx2 = gx[y2, x2]
            vy2 = gy[y2, x2]
            v2norm = math.hypot(vx2, vy2)
            if v2norm < 1e-3:
                continue
            nx2 = vx2 / v2norm
            ny2 = vy2 / v2norm

            dot = nx * nx2 + ny * ny2
            if dot > OPPOSITE_DOT_THRESH:
                continue

            if not line_clear(mask_out, (y, x), (y2, x2)):
                continue

            for t in np.linspace(0.0, 1.0, LINE_SAMPLE_STEPS):
                yy = int(round(y + (y2 - y) * t))
                xx = int(round(x + (x2 - x) * t))
                if 0 <= yy < H and 0 <= xx < W:
                    mask_out[yy, xx] = True

    return mask_out


def compute_mask_and_trunc_info(sem_frame_2d: np.ndarray, target_sem_id: int) -> Dict[str, Any]:
    H, W = sem_frame_2d.shape[:2]
    mask_bool_raw = (sem_frame_2d == target_sem_id)

    if not np.any(mask_bool_raw):
        return {
            "mask_bool": None,
            "mask_solid": None,
            "mask_bin": None,
            "mask_solid_bin": None,
            "mask_xyxy": None,
            "bbox_min_margin_px": None,
            "truncated_closeup": False,
            "mask_area_fraction": 0.0,
            "bbox_area_px": 0,
            "raw_cc_count": 0,
        }

    ys, xs = np.where(mask_bool_raw)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    bbox_w = (x2 - x1 + 1)
    bbox_h = (y2 - y1 + 1)
    bbox_area_px = int(bbox_w * bbox_h)

    margin_left = x1
    margin_top = y1
    margin_right = (W - 1) - x2
    margin_bottom = (H - 1) - y2
    min_margin_px = int(min(margin_left, margin_top, margin_right, margin_bottom))
    truncated_closeup = (min_margin_px <= TRUNC_MARGIN_THRESH_PX)

    mask_area_fraction = float(np.count_nonzero(mask_bool_raw)) / float(H * W)

    mask_bin = np.zeros((H, W), dtype=np.uint8)
    mask_bin[mask_bool_raw] = 255

    # 1) fill enclosed holes
    mask_after_hole = fill_holes(mask_bool_raw)
    # 2) bridge grooves
    mask_after_bridge = fill_concave_gaps_by_gradient(mask_after_hole)
    # 3) fill holes again
    mask_solid_bool = fill_holes(mask_after_bridge)

    mask_solid_bin = np.zeros((H, W), dtype=np.uint8)
    mask_solid_bin[mask_solid_bool] = 255

    raw_cc_count = count_connected_components(mask_bool_raw)

    return {
        "mask_bool": mask_bool_raw,
        "mask_solid": mask_solid_bool,
        "mask_bin": mask_bin,
        "mask_solid_bin": mask_solid_bin,
        "mask_xyxy": [x1, y1, x2, y2],
        "bbox_min_margin_px": min_margin_px,
        "truncated_closeup": bool(truncated_closeup),
        "mask_area_fraction": float(mask_area_fraction),
        "bbox_area_px": bbox_area_px,
        "raw_cc_count": int(raw_cc_count),
    }


def run_dynamic_gates_for_view(
    sim,
    cfg,
    rgb_frame: Optional[np.ndarray],
    sem_2d: Optional[np.ndarray],
    goal_pos: np.ndarray,
    ep_object_category: Optional[str],
    tan_h: float,
    tan_v: float,
) -> Tuple[bool, bool, bool, Dict[str, Any]]:
    """
    Run the three fragile-aware gates on the current view:
    Gate 1: frustum geometry and distance
    Gate 2: mask quality (area, border contact, connected components)
    Gate 3: agreement between the projected goal point and the solid mask, concave shapes included
    Returns: gate1_pass, gate2_pass, gate3_pass, mask_info
    """
    mask_info = {
        "mask_bool": None,
        "mask_solid": None,
        "mask_bin": None,
        "mask_solid_bin": None,
        "mask_xyxy": None,
        "bbox_min_margin_px": None,
        "truncated_closeup": False,
        "mask_area_fraction": 0.0,
        "bbox_area_px": 0,
        "raw_cc_count": 0,
    }

    cam_state = sim.get_agent_state().sensor_states["rgb"]
    cam_pos, fwd, right, up = camera_axes_from_state(cam_state)

    # ---------- Gate 1: frustum geometry ----------
    frustum_ok = frustum_check(
        cam_pos, fwd, right, up,
        tan_h, tan_v,
        goal_pos,
        near=0.05,
        far=100.0,
        eps=1e-6,
    )
    gate1 = bool(frustum_ok)
    gate2 = False
    gate3 = False

    if not gate1:
        return gate1, gate2, gate3, mask_info

    # ---------- Gate 2: mask quality ----------
    if sem_2d is None:
        return gate1, gate2, gate3, mask_info

    mask_info = compute_mask_and_trunc_info(sem_2d, TARGET_SEMANTIC_ID)
    if mask_info["mask_bool"] is None:
        return gate1, gate2, gate3, mask_info

    bbox_area_px = mask_info.get("bbox_area_px", 0)
    truncated_closeup = bool(mask_info["truncated_closeup"])
    raw_cc = int(mask_info.get("raw_cc_count", 0))
    cat_lower = (ep_object_category or "").strip().lower()
    allowed_cc = 2 if (cat_lower == "shoes") else 1

    if truncated_closeup:
        return gate1, gate2, gate3, mask_info
    if bbox_area_px < MIN_AREA_PIXELS:
        return gate1, gate2, gate3, mask_info
    if raw_cc > allowed_cc:
        return gate1, gate2, gate3, mask_info

    gate2 = True

    # ---------- Gate 3: projected goal against the solid mask ----------
    usable_mask_bool = mask_info["mask_solid"]
    if usable_mask_bool is None:
        return gate1, gate2, gate3, mask_info

    if rgb_frame is not None:
        H, W = rgb_frame.shape[:2]
    else:
        H, W = sem_2d.shape[:2]

    cam_q_mn, T_c2w, T_w2c = build_extrinsics(cam_state)
    K = approximate_intrinsics_from_fov(W if W > 0 else 1, H if H > 0 else 1, tan_h, tan_v)
    u_goal, v_goal, z_goal = project_world_point(goal_pos, T_w2c, K)

    in_front = (z_goal < 0.0)
    goal_uv_in_image = (in_front and (0 <= u_goal <= (W - 1)) and (0 <= v_goal <= (H - 1)))

    if not goal_uv_in_image:
        return gate1, gate2, gate3, mask_info

    ys, xs = np.where(usable_mask_bool)
    if xs.size == 0:
        return gate1, gate2, gate3, mask_info

    ur = int(round(u_goal))
    vr = int(round(v_goal))
    ur = max(0, min(W - 1, ur))
    vr = max(0, min(H - 1, vr))

    goal_uv_mask_hit = bool(usable_mask_bool[vr, ur])

    goal_uv_min_dist_px = None
    if not goal_uv_mask_hit:
        du = xs.astype(np.float32) - float(u_goal)
        dv = ys.astype(np.float32) - float(v_goal)
        d = np.sqrt(du * du + dv * dv)
        goal_uv_min_dist_px = float(np.min(d))
    else:
        goal_uv_min_dist_px = 0.0

    near_enough = False
    if goal_uv_mask_hit:
        near_enough = True
    elif goal_uv_min_dist_px is not None and goal_uv_min_dist_px <= GOAL_UV_TOL_PX:
        near_enough = True

    gate3 = bool(near_enough)
    return gate1, gate2, gate3, mask_info


# ============================================================
# Main evaluation
# ============================================================

def evaluate(config_env, args, num_episodes: Optional[int] = None) -> Dict[str, float]:
    """
    Sharded oracle navigation evaluation with gated snapshots, video, and mask visibility stats.
    """
    # === shard the episodes ===
    distributed_env = DistributedEnv(
        config=config_env.habitat,
        num_jobs=args.num_jobs,
        job_index=args.job_index
    )

    env = HabitatPINEnv(distributed_env, config=config_env)
    henv = env.habitat_env  # the underlying Habitat env, which exposes step()

    agent = ShortestPathFollowerAgentPIN(henv.sim, config_env)

    if num_episodes is None:
        num_episodes = len(henv.episodes)
    else:
        assert num_episodes <= len(henv.episodes), (
            f"num_episodes({num_episodes}) > available({len(henv.episodes)})"
        )
    assert num_episodes > 0, "num_episodes should be greater than 0"

    split = config_env.habitat.dataset.split
    results_name = args.exp_name or datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # configurable output root
    output_root = args.output_root
    root_dir = os.path.join(output_root, split, results_name)
    os.makedirs(root_dir, exist_ok=True)

    results_file = os.path.join(root_dir, f"{results_name}_j{args.job_index}_results.jsonl")
    video_dir = os.path.join(root_dir, "videos")
    snapshot_dir = os.path.join(root_dir, "snapshots")

    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)
    if args.save_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)

    wandb_logger = PINDistributedWandbLogger(
        original_num_episodes=distributed_env.original_num_episodes,
        num_jobs=args.num_jobs,
        job_index=args.job_index,
        tmp_dir=root_dir,
        debug=args.debug,
    )

    # precompute the FOV tangents for the gates
    tan_h, tan_v = get_fov_tans_from_cfg(config_env)

    agg_metrics: Dict = defaultdict(float)
    count_episodes = 0

    # gate counters
    per_episode_records = []

    with tqdm(total=num_episodes, desc=f"Worker {args.job_index}/{args.num_jobs}") as pbar:
        while count_episodes < num_episodes:
            observations = env.reset()
            agent.reset()

            pbar.update(1)
            steps = 0
            frames_rgb = []
            frames_mask = []
            
            # record trajectory heights for stair and cross-floor detection
            trajectory_heights = []

            # gate counters for this episode
            gate1_pass = False
            gate2_pass = False
            gate3_pass = False
            gate_attempted = False
            snapshot_rgb_path = None
            snapshot_mask_path = None

            # mask visibility for this episode
            episode_mask_visible = False
            first_mask_frame_index = None
            first_mask_time_sec = None
            first_mask_geodesic_distance = None

            ep = henv.current_episode
            goal_pos = np.array(ep.goals[0].position, dtype=np.float32) if len(ep.goals) > 0 else None

            ep_object_id_str = None
            ep_object_category = None
            try:
                if hasattr(ep, "goals") and len(ep.goals) > 0:
                    g0 = ep.goals[0]
                    if hasattr(g0, "object_id"):
                        ep_object_id_str = str(g0.object_id)
                    if hasattr(g0, "object_category"):
                        ep_object_category = str(g0.object_category)
            except Exception:
                pass
            if (ep_object_id_str is None) or (ep_object_category is None):
                try:
                    if hasattr(ep, "info") and isinstance(ep.info, dict):
                        if ep_object_id_str is None and "object_id" in ep.info:
                            ep_object_id_str = str(ep.info["object_id"])
                        if ep_object_category is None and "object_category" in ep.info:
                            ep_object_category = str(ep.info["object_category"])
                except Exception:
                    pass
            if not ep_object_category:
                ep_object_category = "unlabeled_target"
            if not ep_object_id_str:
                ep_object_id_str = "unlabeled_object"

            # step until the episode ends
            while not env.episode_over:
                current_state = henv.sim.get_agent_state()
                current_pos = np.array(current_state.position, dtype=np.float32)
                
                # record the agent height at this step
                trajectory_heights.append(float(current_pos[1]))

                # geodesic distance, which triggers the gates
                if goal_pos is not None and not gate_attempted:
                    dist_to_goal = henv.sim.geodesic_distance(current_pos, goal_pos)

                    # === within 1 m of the goal: stop, face it, and run the gated snapshot ===
                    if dist_to_goal is not None and dist_to_goal < 1.0:
                        # 1) yaw toward the goal, horizontally
                        direction = goal_pos - current_pos
                        direction[1] = 0.0
                        norm = np.linalg.norm(direction)
                        if norm > 1e-6:
                            direction /= norm
                            quat = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)

                            # normalize again; habitat rejects non-unit quaternions
                            quat = normalize_np_quat(quat)

                            henv.sim.set_agent_state(position=current_pos, rotation=quat)

                        # 2) capture RGB and mask, without tilting
                        snap_obs = henv.sim.get_sensor_observations()
                        rgb_snap = snap_obs.get("rgb", None)
                        sem_arr = snap_obs.get("semantic", snap_obs.get("semantic_sensor", None))
                        if sem_arr is not None and sem_arr.ndim == 3 and sem_arr.shape[-1] == 1:
                            sem_2d = sem_arr[..., 0]
                        else:
                            sem_2d = sem_arr

                        # 3) the three fragile-aware gates
                        g1, g2, g3, mask_info = run_dynamic_gates_for_view(
                            sim=henv.sim,
                            cfg=config_env,
                            rgb_frame=rgb_snap,
                            sem_2d=sem_2d,
                            goal_pos=goal_pos,
                            ep_object_category=ep_object_category,
                            tan_h=tan_h,
                            tan_v=tan_v,
                        )

                        gate1_pass = bool(g1)
                        gate2_pass = bool(g2)
                        gate3_pass = bool(g3)
                        gate_attempted = True

                        # 4) optionally save RGB and the solid mask, under the scene directory
                        if args.save_snapshots:
                            # scene_dir looks like 00877-4ok3usBNeis
                            scene_dir = os.path.basename(os.path.dirname(ep.scene_id))
                            # the basis name, used to disambiguate filenames
                            basis_name = os.path.basename(ep.scene_id).replace(".glb", "")
                            ep_id = ep.episode_id

                            scene_snap_dir = os.path.join(snapshot_dir, scene_dir)
                            ensure_dir(scene_snap_dir)

                            if rgb_snap is not None:
                                if rgb_snap.dtype != np.uint8:
                                    if rgb_snap.dtype in (np.float32, np.float64, np.float16):
                                        rgb_to_save = np.clip(rgb_snap * 255.0, 0, 255).astype(np.uint8)
                                    else:
                                        rgb_to_save = rgb_snap.astype(np.uint8)
                                else:
                                    rgb_to_save = rgb_snap
                                snapshot_rgb_path = os.path.join(
                                    scene_snap_dir, f"{basis_name}_{ep_id}_j{args.job_index}_rgb.png"
                                )
                                imageio.imwrite(snapshot_rgb_path, rgb_to_save)

                            if mask_info is not None and mask_info.get("mask_solid_bin", None) is not None:
                                snapshot_mask_path = os.path.join(
                                    scene_snap_dir, f"{basis_name}_{ep_id}_j{args.job_index}_masksolid.png"
                                )
                                imageio.imwrite(snapshot_mask_path, mask_info["mask_solid_bin"])

                        # this only turns and captures; it does not advance a timestep
                        # the shortest-path agent resumes afterwards

                # one oracle step, including after the gates
                action = agent.act(observations, henv)

                # mask visibility is measured whether or not video is saved
                sem_obs = observations.get("semantic", observations.get("semantic_sensor", None))
                if sem_obs is not None and "rgb" in observations:
                    frame_rgb = observations["rgb"]
                    H, W = frame_rgb.shape[:2]
                    
                    if sem_obs.ndim == 3 and sem_obs.shape[-1] == 1:
                        sem_2d = sem_obs[..., 0]
                    else:
                        sem_2d = sem_obs
                    
                    sem_2d = np.asarray(sem_2d)
                    if sem_2d.shape[0] != H or sem_2d.shape[1] != W:
                        h_min = min(H, sem_2d.shape[0])
                        w_min = min(W, sem_2d.shape[1])
                        sem_2d_cropped = sem_2d[:h_min, :w_min]
                        mask_bool = np.zeros((H, W), dtype=bool)
                        mask_bool[:h_min, :w_min] = (sem_2d_cropped == TARGET_SEMANTIC_ID)
                    else:
                        mask_bool = (sem_2d == TARGET_SEMANTIC_ID)
                    
                    # geodesic distance at which the mask first became visible
                    if mask_bool.any():
                        if not episode_mask_visible:
                            episode_mask_visible = True
                            first_mask_frame_index = steps
                            first_mask_time_sec = steps / float(VIDEO_FPS)
                            
                            if goal_pos is not None:
                                dist_first = henv.sim.geodesic_distance(current_pos, goal_pos)
                                if dist_first is not None:
                                    first_mask_geodesic_distance = float(dist_first)

                # optional video frames: RGB and semantic mask
                if args.save_video and "rgb" in observations:
                    # ----- RGB -----
                    frame_rgb = observations["rgb"]
                    if frame_rgb.dtype != np.uint8:
                        if frame_rgb.dtype in (np.float16, np.float32, np.float64):
                            frame_rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
                        else:
                            frame_rgb = frame_rgb.astype(np.uint8)
                    frames_rgb.append(frame_rgb)

                    # ----- Semantic → Mask Video -----
                    sem_obs_vid = observations.get("semantic", observations.get("semantic_sensor", None))
                    H, W = frame_rgb.shape[:2]

                    if sem_obs_vid is not None:
                        if sem_obs_vid.ndim == 3 and sem_obs_vid.shape[-1] == 1:
                            sem_2d_vid = sem_obs_vid[..., 0]
                        else:
                            sem_2d_vid = sem_obs_vid

                        sem_2d_vid = np.asarray(sem_2d_vid)
                        if sem_2d_vid.shape[0] != H or sem_2d_vid.shape[1] != W:
                            h_min = min(H, sem_2d_vid.shape[0])
                            w_min = min(W, sem_2d_vid.shape[1])
                            sem_2d_vid = sem_2d_vid[:h_min, :w_min]
                            mask_canvas = np.zeros((H, W), dtype=np.uint8)
                            mask_canvas[:h_min, :w_min] = (sem_2d_vid == TARGET_SEMANTIC_ID).astype(np.uint8) * 255
                            mask_img = mask_canvas
                        else:
                            mask_img = np.zeros((H, W), dtype=np.uint8)
                            mask_img[(sem_2d_vid == TARGET_SEMANTIC_ID)] = 255
                    else:
                        # no semantic sensor, emit black
                        mask_img = np.zeros((H, W), dtype=np.uint8)

                    # the mask video is written as 3-channel grey
                    mask_vis = np.stack([mask_img] * 3, axis=-1)
                    frames_mask.append(mask_vis)

                # advance the environment
                observations = henv.step(action)
                steps += 1

            # === episode finished: collect metrics ===
            metrics = {
                k: v for k, v in henv.get_metrics().items()
                if k not in ["top_down_map"]
            }

            ep = henv.current_episode
            metrics["scene_id"] = ep.scene_id
            metrics["episode_id"] = ep.episode_id
            metrics["start_position"] = ep.start_position
            metrics["goal_position"] = ep.goals[0].position if len(ep.goals) > 0 else None

            # record where the agent stopped
            final_state = henv.sim.get_agent_state()
            metrics["final_position"] = [
                float(final_state.position[0]),
                float(final_state.position[1]),
                float(final_state.position[2]),
            ]
            
            # cross-floor analysis, OVON style: path heights plus trajectory heights
            start_pos = ep.start_position
            goal_pos_for_check = ep.goals[0].position if len(ep.goals) > 0 else start_pos
            stair_analysis = analyze_cross_floor(henv.sim, start_pos, goal_pos_for_check, trajectory_heights)
            metrics["stair_analysis"] = {
                "is_cross_floor": stair_analysis["is_cross_floor"],
                "cross_floor_reason": stair_analysis.get("cross_floor_reason"),
                "threshold_used": stair_analysis.get("threshold_used", 0.25),
                # OVON path detection
                "is_cross_floor_path": stair_analysis.get("is_cross_floor_path", False),
                "path_height_range": round(stair_analysis.get("path_height_range", 0.0), 4),
                "path_found": stair_analysis.get("path_found", False),
                # trajectory detection
                "is_cross_floor_trajectory": stair_analysis.get("is_cross_floor_trajectory", False),
                "trajectory_height_range": round(stair_analysis.get("trajectory_height_range", 0.0), 4),
                "height_range": round(stair_analysis.get("height_range", 0.0), 4),
                # heights
                "start_height": round(stair_analysis["start_height"], 4) if stair_analysis.get("start_height") is not None else None,
                "end_height": round(stair_analysis["end_height"], 4) if stair_analysis.get("end_height") is not None else None,
                # kept for backward compatibility
                "detection_method": stair_analysis.get("detection_method", "ovon_pathfinder"),
                "num_floor_clusters": stair_analysis.get("num_floor_clusters", 1),
            }

            spl = float(metrics.get("spl", 0.0))
            cat_spl = float(metrics.get("cat_spl", 0.0))
            pbar.set_description(
                f"W{args.job_index} ep{count_episodes}: len:{metrics.get('episode_length', 0)}, "
                f"s:{metrics.get('success', 0)}, spl:{round(spl, 2)}, cat_spl:{round(cat_spl, 2)}"
            )

            # append the raw navigation metrics
            with jsonlines.open(results_file, mode="a") as f:
                f.write(metrics)

            # wandb record
            wandb_logger.log(metrics)

            # write the videos under the scene directory, RGB and semantic mask
            video_rgb_path = None
            video_mask_path = None
            if args.save_video and len(frames_rgb) > 0:
                scene_dir = os.path.basename(os.path.dirname(ep.scene_id))
                basis_name = os.path.basename(ep.scene_id).replace(".glb", "")
                ep_id = ep.episode_id

                scene_video_dir = os.path.join(video_dir, scene_dir)
                ensure_dir(scene_video_dir)

                video_rgb_path = os.path.join(
                    scene_video_dir, f"{basis_name}_{ep_id}_j{args.job_index}_rgb.mp4"
                )
                imageio.mimsave(video_rgb_path, frames_rgb, fps=VIDEO_FPS)

                if len(frames_mask) == len(frames_rgb):
                    video_mask_path = os.path.join(
                        scene_video_dir, f"{basis_name}_{ep_id}_j{args.job_index}_mask.mp4"
                    )
                    imageio.mimsave(video_mask_path, frames_mask, fps=VIDEO_FPS)

            # running mean
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            agg_metrics[m + "/" + str(sub_m)] += sub_v
                elif isinstance(v, (int, float)):
                    agg_metrics[m] += v

            # per-episode gate and mask-visibility record
            per_episode_records.append({
                "scene_id": metrics["scene_id"],
                "episode_id": metrics["episode_id"],
                "success": float(metrics.get("success", 0.0)),
                "spl": spl,
                "cat_spl": cat_spl,
                "gate_attempted": bool(gate_attempted),
                "gate1_pass": bool(gate1_pass),
                "gate2_pass": bool(gate2_pass),
                "gate3_pass": bool(gate3_pass),
                "snapshot_rgb": snapshot_rgb_path,
                "snapshot_mask": snapshot_mask_path,
                # video paths
                "video_rgb": video_rgb_path,
                "video_mask": video_mask_path,
                # mask visibility
                "episode_mask_visible": bool(episode_mask_visible),
                "first_mask_frame_index": int(first_mask_frame_index) if first_mask_frame_index is not None else None,
                "first_mask_time_sec": float(first_mask_time_sec) if first_mask_time_sec is not None else None,
                "first_mask_geodesic_distance": float(first_mask_geodesic_distance)
                    if first_mask_geodesic_distance is not None else None,
                # cross-floor analysis, OVON style
                "is_cross_floor": stair_analysis["is_cross_floor"],
                "cross_floor_reason": stair_analysis.get("cross_floor_reason"),
                "is_cross_floor_path": stair_analysis.get("is_cross_floor_path", False),
                "is_cross_floor_trajectory": stair_analysis.get("is_cross_floor_trajectory", False),
                "path_height_range": stair_analysis.get("path_height_range", 0.0),
                "trajectory_height_range": stair_analysis.get("trajectory_height_range", 0.0),
                "height_range": stair_analysis.get("height_range", 0.0),
                "threshold_used": stair_analysis.get("threshold_used", 0.25),
                "detection_method": stair_analysis.get("detection_method", "ovon_pathfinder"),
            })

            count_episodes += 1

    # === overall navigation metrics ===
    avg_metrics = {k: v / max(count_episodes, 1) for k, v in agg_metrics.items()}
    print(f"[Worker {args.job_index}] Averages:", avg_metrics)

    # === gate pass rates ===
    num_eps_total = len(per_episode_records)
    num_attempted = sum(1 for r in per_episode_records if r["gate_attempted"])
    num_gate1 = sum(1 for r in per_episode_records if r["gate1_pass"])
    num_gate2 = sum(1 for r in per_episode_records if r["gate2_pass"])
    num_gate3 = sum(1 for r in per_episode_records if r["gate3_pass"])
    num_nav_success = sum(1 for r in per_episode_records if r["success"] > 0.5)

    gate1_rate = float(num_gate1) / float(num_attempted) if num_attempted > 0 else 0.0
    gate2_rate = float(num_gate2) / float(num_attempted) if num_attempted > 0 else 0.0
    gate3_rate = float(num_gate3) / float(num_attempted) if num_attempted > 0 else 0.0
    nav_success_rate = float(num_nav_success) / float(num_eps_total) if num_eps_total > 0 else 0.0

    # === overall mask visibility ===
    num_mask_visible_eps = sum(1 for r in per_episode_records if r["episode_mask_visible"])
    mask_visible_rate = float(num_mask_visible_eps) / float(num_eps_total) if num_eps_total > 0 else 0.0

    # mean geodesic distance at first mask sighting, over episodes that saw one
    first_mask_dists = [
        float(r["first_mask_geodesic_distance"])
        for r in per_episode_records
        if r.get("first_mask_geodesic_distance") is not None
    ]
    avg_first_mask_geodesic_distance = (
        float(sum(first_mask_dists)) / float(len(first_mask_dists))
        if len(first_mask_dists) > 0 else None
    )
    
    # cross-floor statistics
    num_cross_floor_eps = sum(1 for r in per_episode_records if r.get("is_cross_floor", False))
    cross_floor_rate = float(num_cross_floor_eps) / float(num_eps_total) if num_eps_total > 0 else 0.0
    num_with_stairs = sum(1 for r in per_episode_records if r.get("num_stair_segments", 0) > 0)
    stairs_rate = float(num_with_stairs) / float(num_eps_total) if num_eps_total > 0 else 0.0
    avg_stair_steps = (
        sum(r.get("total_stair_steps", 0) for r in per_episode_records) / float(num_eps_total)
        if num_eps_total > 0 else 0.0
    )
    print(f"[Worker {args.job_index}] Cross-floor: {num_cross_floor_eps}/{num_eps_total} ({cross_floor_rate:.2%}), "
          f"With stairs: {num_with_stairs} ({stairs_rate:.2%})")

    # write the JSON summary, including per-episode gate and mask results
    summary = {
        "split": split,
        "exp_name": results_name,
        "num_jobs": args.num_jobs,
        "job_index": args.job_index,
        "num_episodes_total": num_eps_total,
        "num_gate_attempted": num_attempted,
        "nav_success_rate": nav_success_rate,
        "gate1_success_rate": gate1_rate,
        "gate2_success_rate": gate2_rate,
        "gate3_success_rate": gate3_rate,
        "num_mask_visible_episodes": num_mask_visible_eps,
        "mask_visible_rate": mask_visible_rate,
        "avg_first_mask_geodesic_distance": avg_first_mask_geodesic_distance,
        # cross-floor statistics
        "num_cross_floor_episodes": num_cross_floor_eps,
        "cross_floor_rate": cross_floor_rate,
        "num_episodes_with_stairs": num_with_stairs,
        "stairs_rate": stairs_rate,
        "avg_stair_steps_per_episode": round(avg_stair_steps, 2),
        "episodes": per_episode_records,
        "avg_metrics": avg_metrics,
        "generated_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_file = os.path.join(root_dir, f"{results_name}_j{args.job_index}_gate_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Worker {args.job_index}] Gate summary saved to: {summary_file}")

    wandb_logger.close(
        project="pin",
        entity="pin",
        config=dict(config_env),
        name=f"{results_name}_j{args.job_index}"
    )

    return avg_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/pin/pin_hm3d_v1.yaml")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_snapshots", action="store_true", default=False)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "--output_root",
        type=str,
        default=os.environ.get("PIN_RESULT_DIR", "./pin_result"),
        help="Root directory to save eval results",
    )

    args, unknown = parser.parse_known_args()
    args.opts = [o for o in unknown if "=" in o]
    return args


def main():
    args = parse_args()
    config = habitat.get_config(args.config, args.opts)
    print(config)
    evaluate(config_env=config, args=args)


if __name__ == "__main__":
    main()
