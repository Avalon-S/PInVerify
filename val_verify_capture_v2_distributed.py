#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import math
import hashlib
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import quaternion as nq
import random
import datetime

import cv2

import habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors
import magnum as mn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.distributed_env import DistributedEnv
from habitat.envs.habitat_pin_env import HabitatPINEnv


# ============================================================
# Constants
# ============================================================

TARGET_SEMANTIC_ID = 100000

# Episode-level acceptance thresholds
MIN_NAVIGABLE_VIEWPOINTS  = 6    # at least 6 usable viewpoints (navigable and in frustum)
MIN_VALID_MASK_VIEWPOINTS = 3    # at least 3 viewpoints whose mask area clears the category threshold

# Per-category mask-area thresholds in pixels, for 360x640 frames at 42 deg HFOV
CATEGORY_MASK_THRESHOLDS = {
    "keys":        100,
    "watch":       100,
    "eyeglasses":  150,
    "wallet":      150,
    "cellphone":   150,
    "visor":       150,
    "camera":      150,
    "mug":         150,
    "toy":         300,
    "ball":        300,
    "headphones":  300,
    "hat":         300,
    "book":        300,
    "shoes":       300,
    "backpack":    500,
    "bag":         500,
    "laptop":      500,
    "teddy bear":  500,
}
DEFAULT_MASK_THRESHOLD = 200

# object_ids with known rendering defects; their episodes are not captured
SKIP_OBJECT_IDS = {
    "570b82c4391c49ddb1e471e6e55de9f4",  # ball: Adidas Teamgeist, renders solid red because habitat-sim drops its texture
}

# Whether to save the overlay image (RGB with the projected goal point).
# False saves about a third of the image storage; RGB and mask are unaffected.
SAVE_VIS_OVERLAY = False


# ============================================================
# Helpers
# ============================================================

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def to_float_list(x):
    return [float(v) for v in x]

def quat_to_list(q: mn.Quaternion):
    return [float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)]

def npquat_to_magnum(q_np: nq.quaternion) -> mn.Quaternion:
    return mn.Quaternion(
        mn.Vector3(float(q_np.x), float(q_np.y), float(q_np.z)),
        float(q_np.w),
    )

def normalize_np_quat(q: nq.quaternion) -> nq.quaternion:
    n = float(np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z))
    if n <= 1e-12:
        return nq.quaternion(1.0, 0.0, 0.0, 0.0)
    return nq.quaternion(q.w/n, q.x/n, q.y/n, q.z/n)

def mat4_to_list(M: mn.Matrix4):
    return [[float(M[r][c]) for c in range(4)] for r in range(4)]

def vec3_to_list_any(p) -> List[float]:
    if hasattr(p, "x"):
        return [float(p.x), float(p.y), float(p.z)]
    p = np.asarray(p, dtype=np.float32)
    return [float(p[0]), float(p[1]), float(p[2])]

def vec3_to_magnum(p) -> mn.Vector3:
    p = np.asarray(p, dtype=np.float32)
    return mn.Vector3(float(p[0]), float(p[1]), float(p[2]))


def yaw_face_target_np(from_pos: np.ndarray, to_pos: np.ndarray) -> nq.quaternion:
    dir3 = np.asarray(to_pos, dtype=np.float32) - np.asarray(from_pos, dtype=np.float32)
    dir3[1] = 0.0
    n = float(np.linalg.norm(dir3))
    if n < 1e-6:
        return nq.quaternion(1.0, 0.0, 0.0, 0.0)
    dir3 /= n
    q = quat_from_two_vectors(np.array([0.0, 0.0, -1.0], dtype=np.float32), dir3)
    return normalize_np_quat(q)


def camera_axes_from_state(cam_state):
    q = mn.Quaternion(
        mn.Vector3(
            float(cam_state.rotation.x),
            float(cam_state.rotation.y),
            float(cam_state.rotation.z),
        ),
        float(cam_state.rotation.w),
    )
    fwd = np.array(q.transform_vector(mn.Vector3(0,0,-1)), dtype=np.float32)
    right= np.array(q.transform_vector(mn.Vector3(1,0, 0)), dtype=np.float32)
    up   = np.array(q.transform_vector(mn.Vector3(0,1, 0)), dtype=np.float32)
    pos  = np.array(cam_state.position, dtype=np.float32)
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


def get_camera_intrinsics_from_cfg(cfg) -> dict:
    try:
        rgb_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    except Exception:
        rgb_cfg = cfg.simulator.agents.main_agent.sim_sensors.rgb_sensor

    width, height = int(rgb_cfg.width), int(rgb_cfg.height)
    hfov_deg = float(rgb_cfg.hfov)
    if hasattr(rgb_cfg, "vfov") and float(getattr(rgb_cfg, "vfov", 0.0)) > 0:
        vfov_deg = float(rgb_cfg.vfov)
    else:
        vfov_deg = math.degrees(
            2 * math.atan(
                math.tan(math.radians(hfov_deg)/2.0) * (height/width)
            )
        )

    fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    fy = 0.5 * height / math.tan(math.radians(vfov_deg) / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return {
        "width": width,
        "height": height,
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
    }


def get_fov_tans_from_cfg(cfg) -> Tuple[float, float]:
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
                math.tan(math.radians(hfov)/2.0) * (H/W)
            )
        )
    return math.tan(math.radians(hfov)*0.5), math.tan(math.radians(vfov)*0.5)


def get_rgb_sensor_offset_y(cfg, default_y: float = 1.31) -> float:
    try:
        pos = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position
    except Exception:
        try:
            pos = cfg.simulator.agents.main_agent.sim_sensors.rgb_sensor.position
        except Exception:
            pos = None
    if pos is None:
        return default_y
    try:
        return float(pos[1])
    except Exception:
        return default_y


# ============================================================
# runtime goal information
# ============================================================

def get_target_object_info(sim: habitat_sim.Simulator) -> Optional[Dict[str, Any]]:
    rom = sim.get_rigid_object_manager()

    if hasattr(rom, "get_objects_by_handle_substring"):
        objs_dict = rom.get_objects_by_handle_substring("")
        for _h, robj in objs_dict.items():
            sem_id = getattr(robj, "semantic_id", None)
            if sem_id == TARGET_SEMANTIC_ID:
                return {
                    "object_id_runtime_int": int(robj.object_id),
                    "translation": np.array(robj.translation, dtype=np.float32),
                }

    if hasattr(rom, "get_object_handles") and hasattr(rom, "get_object_by_handle"):
        try:
            for hndl in rom.get_object_handles():
                robj = rom.get_object_by_handle(hndl)
                if robj is None:
                    continue
                sem_id = getattr(robj, "semantic_id", None)
                if sem_id == TARGET_SEMANTIC_ID:
                    return {
                        "object_id_runtime_int": int(robj.object_id),
                        "translation": np.array(robj.translation, dtype=np.float32),
                    }
        except Exception:
            pass

    return None


# ============================================================
# camera extrinsics and projection
# ============================================================

def build_extrinsics(cam_state):
    cam_q_np = normalize_np_quat(cam_state.rotation)
    cam_q_mn = npquat_to_magnum(cam_q_np)
    cam_p_mn = vec3_to_magnum(cam_state.position)
    R3 = cam_q_mn.to_matrix()
    T_c2w = mn.Matrix4.from_(R3, cam_p_mn)
    T_w2c = T_c2w.inverted()
    return cam_q_mn, T_c2w, T_w2c


def approximate_intrinsics_from_fov(W:int, H:int, tan_h:float, tan_v:float) -> np.ndarray:
    fx = 0.5 * W / tan_h
    fy = 0.5 * H / tan_v
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)


def project_world_point(pt_world: np.ndarray, T_w2c: mn.Matrix4, K: np.ndarray) -> Tuple[float, float, float]:
    pc = T_w2c.transform_point(mn.Vector3(float(pt_world[0]), float(pt_world[1]), float(pt_world[2])))
    z = float(pc.z)
    z_safe = z if abs(z) > 1e-6 else 1e-6
    u = float(K[0,0] * (pc.x / z_safe) + K[0,2])
    v = float(K[1,1] * (pc.y / z_safe) + K[1,2])
    return u, v, z


# ============================================================
# Mask handling
# ============================================================

def compute_mask_info(sem_frame_2d: np.ndarray, target_sem_id: int) -> Dict[str, Any]:
    """Extract the raw mask, its bbox and area. No hole filling or connected-component analysis.
    mask_bin is built lazily, only when it actually has to be written."""
    H, W = sem_frame_2d.shape[:2]
    mask_bool = (sem_frame_2d == target_sem_id)

    if not np.any(mask_bool):
        return {
            "has_mask": False,
            "mask_bool": None,
            "mask_area_px": 0,
            "mask_bbox_xyxy": None,
            "mask_area_fraction": 0.0,
        }

    ys, xs = np.where(mask_bool)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    mask_area_px = int(len(ys))  # same as count_nonzero, but ys is already computed

    return {
        "has_mask": True,
        "mask_bool": mask_bool,   # lazy: converted to uint8 only when saved
        "mask_area_px": mask_area_px,
        "mask_bbox_xyxy": [x1, y1, x2, y2],
        "mask_area_fraction": float(mask_area_px) / float(H * W),
    }


# ============================================================
# Overview image
# ============================================================

def save_overview_plot(save_path: str, goal_pos: np.ndarray, viewpoints: List[dict],
                       arrow_len: float = 0.25):
    """
    Top-down overview with three-color viewpoint markers:
      Green  = mask meets threshold (valid)
      Orange = navigable but mask not meeting threshold
      Red X  = not navigable / frustum failed
    """
    from matplotlib.lines import Line2D

    gx, gz = float(goal_pos[0]), float(goal_pos[2])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter([0], [0], marker='*', s=160, c='gold', edgecolors='k', zorder=5)

    colors_cfg = [
        ("valid",   "green",  True),
        ("no_mask", "orange", True),
        ("failed",  "red",    False),
    ]

    for quality, color, draw_arrow in colors_cfg:
        for range_label, mkr in [("near", "o"), ("far", "s")]:
            xs, zs, us, vs = [], [], [], []

            for vp in viewpoints:
                nav = vp.get("navigable", False)
                meets = vp.get("mask_meets_threshold", False)
                vp_q = "valid" if (nav and meets) else ("no_mask" if nav else "failed")
                if vp_q != quality:
                    continue
                if vp.get("range_label") != range_label:
                    continue

                cam = vp.get("camera_position")
                if cam is None:
                    continue

                xs.append(cam[0] - gx)
                zs.append(cam[2] - gz)

                fwd = vp.get("camera_fwd_xz")
                if fwd and draw_arrow:
                    fx, fz = fwd
                    n = math.hypot(fx, fz) + 1e-12
                    us.append(fx / n * arrow_len)
                    vs.append(fz / n * arrow_len)
                else:
                    us.append(0)
                    vs.append(0)

            if not xs:
                continue

            use_mkr = "x" if quality == "failed" else mkr
            ax.scatter(xs, zs, marker=use_mkr,
                       s=(50 if quality == "failed" else 30),
                       c=color, alpha=0.9)
            if draw_arrow:
                ax.quiver(xs, zs, us, vs, angles='xy', scale_units='xy',
                          scale=1.0, width=0.005, color=color, alpha=0.8)

    n_valid   = sum(1 for v in viewpoints if v.get("navigable") and v.get("mask_meets_threshold"))
    n_no_mask = sum(1 for v in viewpoints if v.get("navigable") and not v.get("mask_meets_threshold"))
    n_failed  = sum(1 for v in viewpoints if not v.get("navigable"))
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='k', markersize=10, label='goal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=8, label=f'valid mask ({n_valid})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label=f'no mask ({n_no_mask})'),
        Line2D([0], [0], marker='x', color='red',
               markersize=8, label=f'failed ({n_failed})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('x (goal-centered, m)')
    ax.set_ylabel('z (goal-centered, m)')
    ax.set_title('Camera poses around goal (top-down)')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# Pose sampling
# ============================================================

def sector_angle_ranges(n_sectors: int = 12, start_deg: float = 0.0) -> List[Tuple[float,float]]:
    step = 360.0 / n_sectors
    return [(start_deg + i*step, start_deg + (i+1)*step) for i in range(n_sectors)]

def sample_on_ring_around_goal(env, goal_pos: np.ndarray, radius: float,
                               theta_deg_rng: Tuple[float,float], max_tries: int = 50) -> Optional[np.ndarray]:
    pf = env.sim.pathfinder
    for _ in range(max_tries):
        th = np.deg2rad(np.random.uniform(theta_deg_rng[0], theta_deg_rng[1]))
        x = goal_pos[0] + radius * math.cos(th)
        z = goal_pos[2] + radius * math.sin(th)
        snapped = pf.snap_point(np.array([x, goal_pos[1], z], dtype=np.float32))
        if not pf.is_navigable(snapped):
            continue
        return np.asarray(snapped, dtype=np.float32)
    return None

def sample_on_ray_at_theta(env, goal_pos: np.ndarray, radius: float, theta_rad: float,
                           max_tries: int = 15, perturb_deg: float = 2.0) -> Optional[np.ndarray]:
    pf = env.sim.pathfinder
    for k in range(max_tries):
        dtheta = 0.0 if k == 0 else math.radians(np.random.uniform(-perturb_deg, perturb_deg))
        th = theta_rad + dtheta
        x = goal_pos[0] + radius * math.cos(th)
        z = goal_pos[2] + radius * math.sin(th)
        snapped = pf.snap_point(np.array([x, goal_pos[1], z], dtype=np.float32))
        if not pf.is_navigable(snapped):
            continue
        return np.asarray(snapped, dtype=np.float32)
    return None

def polar_from_goal(cam_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[float, float]:
    dx = float(cam_pos[0] - goal_pos[0])
    dz = float(cam_pos[2] - goal_pos[2])
    radius_m = math.hypot(dx, dz)
    theta_deg = math.degrees(math.atan2(dz, dx))
    if theta_deg < 0.0:
        theta_deg += 360.0
    return radius_m, theta_deg

def in_sector(theta_deg: float, sector_deg_rng: Tuple[float,float]) -> bool:
    ang0, ang1 = sector_deg_rng
    return (theta_deg >= ang0) and (theta_deg <= ang1)

def in_radius(radius_m: float, rmin: float, rmax: float) -> bool:
    return (radius_m >= rmin) and (radius_m <= rmax)


# ============================================================
# Viewpoint capture
# ============================================================

def make_failed_viewpoint(tag: str, sector_index: int, range_label: str) -> dict:
    """Placeholder record for a viewpoint with no navigable, in-frustum position."""
    return {
        "tag": tag,
        "navigable": False,
        "in_frustum": False,
        "has_mask": False,
        "mask_area_px": 0,
        "mask_bbox_xyxy": None,
        "mask_area_fraction": 0.0,
        "mask_meets_threshold": False,
        "category_threshold_used": None,
        "rgb": None,
        "mask_raw_path": None,
        "camera_position": None,
        "camera_rotation_quat_wxyz": None,
        "camera_to_world": None,
        "world_to_camera": None,
        "camera_fwd_xz": None,
        "goal_position": None,
        "goal_position_mode": None,
        "target_object_id_runtime_int": None,
        "episode_object_id_str": None,
        "episode_object_category": None,
        "floor_y_at_cam": None,
        "base_y_used": None,
        "base_y_source": None,
        "cam_y_world": None,
        "action": None,
        "sector_index": sector_index,
        "range_label": range_label,
        "radius_m": None,
        "angle_deg_range": None,
    }


def capture_viewpoint(
    henv,
    base_pos_nav: np.ndarray,
    base_y_override: Optional[float],
    sensor_offset_y: float,
    tan_h: float,
    tan_v: float,
    look_thresh: float,
    out_rgb_dir: str,
    out_mask_dir: str,
    out_vis_dir: str,
    tag: str,
    base_y_source: str,
    ep_object_id_str: Optional[str],
    ep_object_category: Optional[str],
    category_threshold: int,
    save_vis: bool,
) -> Optional[dict]:
    """
    Capture one frame at the given position and return its quality record.
    Returns None if the frustum check fails, in which case the caller retries elsewhere.
    """
    ep = henv.current_episode

    # ---- goal information ----
    tgt_obj = get_target_object_info(henv.sim)
    if tgt_obj is not None:
        tgt_world_pos = np.array(tgt_obj["translation"], dtype=np.float32)
        tgt_object_id_runtime_int = int(tgt_obj["object_id_runtime_int"])
        goal_position_mode = "runtime"
    else:
        tgt_world_pos = np.array(ep.goals[0].position, dtype=np.float32)
        tgt_object_id_runtime_int = None
        goal_position_mode = "canonical"

    # ---- place the agent and face the goal ----
    base_y = float(base_pos_nav[1]) if base_y_override is None else float(base_y_override)
    agent_base_pos = np.array([float(base_pos_nav[0]), base_y, float(base_pos_nav[2])],
                              dtype=np.float32)
    yaw_np_quat = yaw_face_target_np(agent_base_pos, tgt_world_pos)
    henv.sim.set_agent_state(position=mn.Vector3(*agent_base_pos), rotation=yaw_np_quat)

    # ---- pitch adjustment ----
    cam_y_world = float(base_y + sensor_offset_y)
    hdiff = float(tgt_world_pos[1] - cam_y_world)
    action_tag = "none"
    if hdiff > look_thresh:
        action_tag = "look_up"
        try:
            henv.step({"action": "look_up"})
        except Exception:
            pass
    elif hdiff < -look_thresh:
        action_tag = "look_down"
        try:
            henv.step({"action": "look_down"})
        except Exception:
            pass

    # ---- Frustum check ----
    cam_state = henv.sim.get_agent_state().sensor_states["rgb"]
    cam_pos, fwd, right, up = camera_axes_from_state(cam_state)

    frustum_ok = frustum_check(
        cam_pos, fwd, right, up,
        tan_h, tan_v,
        tgt_world_pos,
        near=0.05, far=100.0, eps=1e-6,
    )

    if not frustum_ok:
        return None  # the caller should retry another position

    # ---- read the observation ----
    obs = henv.sim.get_sensor_observations()
    rgb_frame = obs.get("rgb", None)
    sem_arr = obs.get("semantic", obs.get("semantic_sensor", None))

    if sem_arr is not None and sem_arr.ndim == 3 and sem_arr.shape[-1] == 1:
        sem_2d = sem_arr[..., 0]
    else:
        sem_2d = sem_arr

    if rgb_frame is not None:
        H, W = rgb_frame.shape[:2]
    elif sem_2d is not None:
        H, W = sem_2d.shape[:2]
    else:
        H, W = -1, -1

    # ---- mask analysis ----
    if sem_2d is not None:
        mask_info = compute_mask_info(sem_2d, TARGET_SEMANTIC_ID)
    else:
        mask_info = {
            "has_mask": False,
            "mask_bin": None,
            "mask_area_px": 0,
            "mask_bbox_xyxy": None,
            "mask_area_fraction": 0.0,
        }

    has_mask = mask_info["has_mask"]
    mask_area_px = mask_info["mask_area_px"]
    mask_meets_threshold = (mask_area_px >= category_threshold)

    # ---- extrinsics, reusing the cam_state from the frustum check ----
    cam_q_mn, T_c2w, T_w2c = build_extrinsics(cam_state)

    # ---- write images ----
    rgb_name = f"rgb_{tag}.png"
    mask_name = f"mask_{tag}.png"
    rgb_rel = os.path.join("rgb", rgb_name)
    mask_rel = os.path.join("mask", mask_name) if has_mask else None

    if save_vis:
        # always save RGB; PNG is lossless and cv2 is 2-3x faster than imageio here
        if rgb_frame is not None:
            cv2.imwrite(
                os.path.join(out_rgb_dir, rgb_name),
                cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
            )
        # save the mask only when there is one
        if has_mask and mask_info["mask_bool"] is not None:
            mask_bin = np.zeros_like(mask_info["mask_bool"], dtype=np.uint8)
            mask_bin[mask_info["mask_bool"]] = 255
            cv2.imwrite(os.path.join(out_mask_dir, mask_name), mask_bin)
        # overlay: RGB plus the projected goal dot, drawn with OpenCV (~100x faster than matplotlib)
        if SAVE_VIS_OVERLAY and rgb_frame is not None and H > 0 and W > 0:
            try:
                K = approximate_intrinsics_from_fov(W, H, tan_h, tan_v)
                u_goal, v_goal, z_goal = project_world_point(tgt_world_pos, T_w2c, K)
                vis_name = f"vis_{tag}.png"
                vis_img = rgb_frame.copy()
                if z_goal < 0.0:
                    cx, cy = int(round(u_goal)), int(round(v_goal))
                    cv2.circle(vis_img, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.circle(vis_img, (cx, cy), 6, (255, 255, 255), 1)
                cv2.imwrite(
                    os.path.join(out_vis_dir, vis_name),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                )
            except Exception:
                pass

    cam_p_list = vec3_to_list_any(cam_state.position)
    fwd_vec = cam_q_mn.transform_vector(mn.Vector3(0.0, 0.0, -1.0))

    record = {
        "tag": tag,
        "navigable": True,
        "in_frustum": True,
        "has_mask": has_mask,
        "mask_area_px": int(mask_area_px),
        "mask_bbox_xyxy": mask_info["mask_bbox_xyxy"],
        "mask_area_fraction": mask_info["mask_area_fraction"],
        "mask_meets_threshold": mask_meets_threshold,
        "category_threshold_used": int(category_threshold),
        "rgb": rgb_rel,
        "mask_raw_path": mask_rel,
        "camera_position": cam_p_list,
        "camera_rotation_quat_wxyz": quat_to_list(cam_q_mn),
        "camera_to_world": mat4_to_list(T_c2w),
        "world_to_camera": mat4_to_list(T_w2c),
        "camera_fwd_xz": [float(fwd_vec.x), float(fwd_vec.z)],
        "goal_position": to_float_list(tgt_world_pos),
        "goal_position_mode": goal_position_mode,
        "target_object_id_runtime_int": tgt_object_id_runtime_int,
        "episode_object_id_str": ep_object_id_str,
        "episode_object_category": ep_object_category,
        "floor_y_at_cam": float(agent_base_pos[1]),
        "base_y_used": float(agent_base_pos[1]),
        "base_y_source": base_y_source,
        "cam_y_world": float(cam_p_list[1]),
        "action": action_tag,
    }

    return record


# ============================================================
# Main loop
# ============================================================

def collect_dataset(
    config_env,
    output_root: str,
    n_sectors: int,
    near_min: float,
    near_max: float,
    far_min: float,
    far_max: float,
    sector_skip: int,
    look_thresh: float,
    max_tries: int,
    num_jobs: int,
    job_index: int,
    debug: bool,
    seed: int,
    save_vis: bool,
):
    seed_eff = int(seed) + int(job_index)
    np.random.seed(seed_eff)
    random.seed(seed_eff)

    distributed_env = DistributedEnv(
        config=config_env.habitat,
        num_jobs=num_jobs,
        job_index=job_index,
    )
    pin_env = HabitatPINEnv(distributed_env, config=config_env)
    henv = pin_env.habitat_env

    intrinsics = get_camera_intrinsics_from_cfg(config_env)
    tan_h, tan_v = get_fov_tans_from_cfg(config_env)
    sensor_offset_y = get_rgb_sensor_offset_y(config_env, default_y=1.31)

    ensure_dir(output_root)

    total_eps = len(henv.episodes)
    print(f"[INFO] shard {job_index}/{num_jobs} | episodes={total_eps} "
          f"| seed={seed_eff} | save_vis={save_vis}")

    sector_ranges_all = sector_angle_ranges(n_sectors=n_sectors, start_deg=0.0)
    sector_order = list(range(0, n_sectors, sector_skip + 1))

    for _ in range(total_eps):
        henv.reset()
        ep = henv.current_episode

        nominal_goal_pos = np.array(ep.goals[0].position, dtype=np.float32)

        # ---- fixed per-episode random seed ----
        # The base seed excludes job_index, so an episode samples the same viewpoints
        # no matter which job picks it up or which phase captures it.
        _ep_seed_str = f"{ep.scene_id}_{ep.episode_id}_{seed}"
        _ep_hash = int(hashlib.md5(_ep_seed_str.encode()).hexdigest(), 16) % (2**31)
        np.random.seed(_ep_hash)
        random.seed(_ep_hash)

        # ---- object information ----
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

        # ---- skip object_ids with known rendering defects ----
        if ep_object_id_str in SKIP_OBJECT_IDS:
            print(f"  [SKIP] episode {getattr(ep, 'episode_id', '?')}: "
                  f"object_id={ep_object_id_str} in SKIP_OBJECT_IDS")
            continue

        # ---- category threshold ----
        category_threshold = CATEGORY_MASK_THRESHOLDS.get(
            ep_object_category.strip().lower(), DEFAULT_MASK_THRESHOLD
        )

        scene_key = scene_key_from_scene_id(ep.scene_id)
        episode_id = (getattr(ep, "episode_id", None)
                      or getattr(ep, "episodeId", None)
                      or f"episode_{getattr(ep, 'episode_index', 'unknown')}")

        base_dir = os.path.join(output_root, scene_key, str(episode_id))
        rgb_dir = os.path.join(base_dir, "rgb")
        mask_dir = os.path.join(base_dir, "mask")
        vis_dir = os.path.join(base_dir, "vis")
        ensure_dir(base_dir)
        # create the subdirectories up front so capture_viewpoint need not
        if save_vis:
            ensure_dir(rgb_dir)
            ensure_dir(mask_dir)
            if SAVE_VIS_OVERLAY:
                ensure_dir(vis_dir)

        pf = henv.sim.pathfinder
        temp_snap = pf.snap_point(nominal_goal_pos)
        if pf.is_navigable(temp_snap):
            final_base_y = float(temp_snap[1])
            base_y_source = "snap_goal_y"
        else:
            final_base_y = float(nominal_goal_pos[1])
            base_y_source = "goal_y_raw"

        viewpoints: List[dict] = []
        far_theta_by_sec: Dict[int, float] = {}

        for sec_idx in sector_order:
            ang0, ang1 = sector_ranges_all[sec_idx]

            # ---- far-ring viewpoints ----
            far_record = None
            consec_nav_fail = 0   # consecutive non-navigable samples
            for _attempt in range(max_tries):
                sample_radius = float(np.random.uniform(far_min, far_max))
                cand_nav = sample_on_ring_around_goal(
                    henv, nominal_goal_pos, sample_radius, (ang0, ang1), max_tries=1,
                )
                if cand_nav is None:
                    consec_nav_fail += 1
                    if consec_nav_fail >= 15:   # give up on this sector after 15 consecutive non-navigable samples
                        break
                    continue
                consec_nav_fail = 0

                rad_now, th_now = polar_from_goal(cand_nav, nominal_goal_pos)
                if (not in_sector(th_now, (ang0, ang1))
                        or not in_radius(rad_now, far_min, far_max)):
                    continue

                result = capture_viewpoint(
                    henv=henv,
                    base_pos_nav=cand_nav,
                    base_y_override=final_base_y,
                    sensor_offset_y=sensor_offset_y,
                    tan_h=tan_h, tan_v=tan_v,
                    look_thresh=look_thresh,
                    out_rgb_dir=rgb_dir,
                    out_mask_dir=mask_dir,
                    out_vis_dir=vis_dir,
                    tag=f"s{sec_idx}_far",
                    base_y_source=base_y_source,
                    ep_object_id_str=ep_object_id_str,
                    ep_object_category=ep_object_category,
                    category_threshold=category_threshold,
                    save_vis=save_vis,
                )

                if result is not None:
                    result["sector_index"] = int(sec_idx)
                    result["angle_deg_range"] = [float(ang0), float(ang1)]
                    result["range_label"] = "far"
                    result["radius_m"] = float(rad_now)
                    far_record = result

                    cam_px = result["camera_position"][0]
                    cam_pz = result["camera_position"][2]
                    far_theta_by_sec[int(sec_idx)] = float(
                        math.atan2(cam_pz - nominal_goal_pos[2],
                                   cam_px - nominal_goal_pos[0])
                    )
                    break

            if far_record is None:
                far_record = make_failed_viewpoint(f"s{sec_idx}_far", sec_idx, "far")
            viewpoints.append(far_record)

            # ---- near-ring viewpoints ----
            near_record = None
            consec_nav_fail = 0
            for _attempt in range(max_tries):
                sample_radius = float(np.random.uniform(near_min, near_max))

                cand_nav_near = None
                if int(sec_idx) in far_theta_by_sec:
                    theta_target = far_theta_by_sec[int(sec_idx)]
                    cand_nav_near = sample_on_ray_at_theta(
                        henv, nominal_goal_pos, sample_radius,
                        theta_rad=theta_target, max_tries=3, perturb_deg=2.0,
                    )
                if cand_nav_near is None:
                    cand_nav_near = sample_on_ring_around_goal(
                        henv, nominal_goal_pos, sample_radius, (ang0, ang1), max_tries=1,
                    )
                if cand_nav_near is None:
                    consec_nav_fail += 1
                    if consec_nav_fail >= 15:
                        break
                    continue
                consec_nav_fail = 0

                rad_now, th_now = polar_from_goal(cand_nav_near, nominal_goal_pos)
                if (not in_sector(th_now, (ang0, ang1))
                        or not in_radius(rad_now, near_min, near_max)):
                    continue

                result = capture_viewpoint(
                    henv=henv,
                    base_pos_nav=cand_nav_near,
                    base_y_override=final_base_y,
                    sensor_offset_y=sensor_offset_y,
                    tan_h=tan_h, tan_v=tan_v,
                    look_thresh=look_thresh,
                    out_rgb_dir=rgb_dir,
                    out_mask_dir=mask_dir,
                    out_vis_dir=vis_dir,
                    tag=f"s{sec_idx}_near",
                    base_y_source=base_y_source,
                    ep_object_id_str=ep_object_id_str,
                    ep_object_category=ep_object_category,
                    category_threshold=category_threshold,
                    save_vis=save_vis,
                )

                if result is not None:
                    result["sector_index"] = int(sec_idx)
                    result["angle_deg_range"] = [float(ang0), float(ang1)]
                    result["range_label"] = "near"
                    result["radius_m"] = float(rad_now)
                    near_record = result
                    break

            if near_record is None:
                near_record = make_failed_viewpoint(f"s{sec_idx}_near", sec_idx, "near")
            viewpoints.append(near_record)

        # ---- episode-level verdict ----
        navigable_count = sum(1 for v in viewpoints if v.get("navigable", False))
        valid_mask_count = sum(1 for v in viewpoints if v.get("mask_meets_threshold", False))
        episode_success = (
            navigable_count >= MIN_NAVIGABLE_VIEWPOINTS
            and valid_mask_count >= MIN_VALID_MASK_VIEWPOINTS
        )

        # ---- write meta.json ----
        meta = {
            "scene_id": ep.scene_id,
            "scene_key": scene_key,
            "episode_id": episode_id,
            "goal_position_nominal": to_float_list(nominal_goal_pos.tolist()),
            "final_base_y": float(final_base_y),
            "base_y_source": base_y_source,
            "object_category": ep_object_category,
            "object_id": ep_object_id_str,
            "pin_semantic_id_convention": {
                "target_semantic_id": TARGET_SEMANTIC_ID,
                "meaning": ("All pixels with this semantic_id correspond to "
                            "this episode's target object."),
            },
            "camera_intrinsics": intrinsics,
            "sensor_mount_offset_y": float(sensor_offset_y),
            "n_sectors": n_sectors,
            "sector_order": sector_order,
            "sector_skip": int(sector_skip),
            "ranges": {
                "near": [float(near_min), float(near_max)],
                "far":  [float(far_min),  float(far_max)],
            },
            "look_thresh": float(look_thresh),
            "category_mask_threshold": category_threshold,
            "viewpoints": viewpoints,
            "episode_result": {
                "navigable_count": navigable_count,
                "valid_mask_count": valid_mask_count,
                "total_viewpoints": len(viewpoints),
                "success": episode_success,
                "object_category": ep_object_category,
            },
            "captures": [v for v in viewpoints if v.get("navigable", False)],
            "overview": None,
            "regen_info": {
                "generated_by": "val_verify_capture_v2_distributed.py (episode-level eval)",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # ---- overview image ----
        if save_vis:
            overview_rel = "overview.png"
            overview_path = os.path.join(base_dir, overview_rel)
            try:
                save_overview_plot(overview_path, nominal_goal_pos,
                                   viewpoints, arrow_len=0.25)
                meta["overview"] = overview_rel
            except Exception as e:
                print(f"[WARN] overview plot failed {scene_key}/{episode_id}: {e}")

        meta["regen_info"]["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(base_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = "OK" if episode_success else "FAIL"
        print(f"[{status}] {scene_key}/{episode_id}: "
              f"nav={navigable_count} mask_valid={valid_mask_count} "
              f"| threshold={category_threshold}px ({ep_object_category})")


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="configs/models/pin/pin_hm3d_v1.yaml")
    ap.add_argument("--output_root", type=str,
                    default="./captures/pin_capture_active")

    ap.add_argument("--n_sectors", type=int, default=12)
    ap.add_argument("--near_min", type=float, default=0.9)
    ap.add_argument("--near_max", type=float, default=1.2)
    ap.add_argument("--far_min", type=float, default=1.4)
    ap.add_argument("--far_max", type=float, default=1.7)
    ap.add_argument("--sector_skip", type=int, default=1)

    ap.add_argument("--look_thresh", type=float, default=0.3)
    ap.add_argument("--max_tries", type=int, default=50)
    ap.add_argument("--num_jobs", type=int, default=1)
    ap.add_argument("--job_index", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_vis", action="store_true")

    args, unknown = ap.parse_known_args()
    args.opts = [o for o in unknown if "=" in o]
    return args


def scene_key_from_scene_id(scene_id: str) -> str:
    parts = scene_id.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(scene_id))[0]


def main():
    args = parse_args()
    cfg = habitat.get_config(args.config, args.opts)

    collect_dataset(
        config_env=cfg,
        output_root=args.output_root,
        n_sectors=args.n_sectors,
        near_min=args.near_min,
        near_max=args.near_max,
        far_min=args.far_min,
        far_max=args.far_max,
        sector_skip=args.sector_skip,
        look_thresh=args.look_thresh,
        max_tries=args.max_tries,
        num_jobs=args.num_jobs,
        job_index=args.job_index,
        debug=args.debug,
        seed=args.seed,
        save_vis=args.save_vis,
    )


if __name__ == "__main__":
    main()
