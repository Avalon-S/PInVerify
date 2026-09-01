#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floor detection from semantic annotations.

Reads real floor heights from an HM3D scene's semantic annotations, so a position
can be assigned to a floor exactly rather than by a height threshold.

Usage:
    from semantic_floor_detector import SemanticFloorDetector
    
    detector = SemanticFloorDetector(sim)
    floor_id = detector.get_floor_at_height(position_y)
    is_cross_floor = detector.check_cross_floor(start_pos, goal_pos)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class SemanticFloorDetector:
    """Floor detector backed by semantic annotations."""
    
    def __init__(self, sim, floor_keywords: List[str] = None):
        """
        Set up the detector.
        
        Args:
            sim: the Habitat simulator
            floor_keywords: category names that count as floor, default ["floor"]
        """
        self.sim = sim
        self.floor_keywords = floor_keywords or ["floor"]
        
        # cached floor information
        self._floor_heights: Dict[int, float] = {}  # floor_id -> floor_y
        self._floor_ranges: Dict[int, Tuple[float, float]] = {}  # floor_id -> (min_y, max_y)
        self._sorted_floors: List[Tuple[int, float]] = []  # [(floor_id, floor_y), ...] sorted by height
        self._initialized = False
        
        # initialize
        self._initialize()
    
    def _initialize(self):
        """Pull floor information out of the semantic annotations."""
        try:
            sem_scene = self.sim.semantic_annotations()
            if sem_scene is None:
                print("[SemanticFloorDetector] Warning: No semantic annotations available")
                return
            
            floor_objects_by_level = defaultdict(list)
            all_categories_found = set()
            
            # debug output
            print(f"[SemanticFloorDetector] sem_scene type: {type(sem_scene)}")
            
            # try the objects list
            if hasattr(sem_scene, 'objects'):
                objs = sem_scene.objects
                print(f"[SemanticFloorDetector] objects type: {type(objs)}")
                try:
                    print(f"[SemanticFloorDetector] objects len: {len(objs)}")
                except:
                    print("[SemanticFloorDetector] Cannot get len of objects")
            
            # try the levels list
            if hasattr(sem_scene, 'levels'):
                lvls = sem_scene.levels
                print(f"[SemanticFloorDetector] levels type: {type(lvls)}")
                try:
                    print(f"[SemanticFloorDetector] levels len: {len(lvls)}")
                except:
                    print("[SemanticFloorDetector] Cannot get len of levels")
            
            # helper: category name
            def get_category_name(obj):
                try:
                    if hasattr(obj, 'category') and obj.category is not None:
                        cat = obj.category
                        if hasattr(cat, 'name'):
                            name = cat.name
                            return str(name()) if callable(name) else str(name)
                except:
                    pass
                return ""
            
            # walk objects by index
            if hasattr(sem_scene, 'objects'):
                objs = sem_scene.objects
                try:
                    obj_len = len(objs)
                    print(f"[SemanticFloorDetector] Iterating {obj_len} objects by index...")
                    
                    for i in range(obj_len):
                        try:
                            obj = objs[i]
                            if obj is None:
                                continue
                            
                            category_name = get_category_name(obj)
                            if category_name:
                                all_categories_found.add(category_name)
                            
                            # is this a floor
                            if any(kw in category_name.lower() for kw in self.floor_keywords):
                                if hasattr(obj, 'aabb') and obj.aabb is not None:
                                    center = obj.aabb.center
                                    floor_y = float(center[1])
                                    
                                    # floor id
                                    level_id = 0
                                    if hasattr(obj, 'region') and obj.region is not None:
                                        if hasattr(obj.region, 'level') and obj.region.level is not None:
                                            try:
                                                level_id = int(obj.region.level.id)
                                            except:
                                                pass
                                    
                                    floor_objects_by_level[level_id].append(floor_y)
                        except Exception as e:
                            # skip invalid indices
                            continue
                except Exception as e:
                    print(f"[SemanticFloorDetector] Error iterating objects: {e}")
            
            # report the categories found
            if len(all_categories_found) > 0:
                floor_like = [c for c in all_categories_found if c and "floor" in c.lower()]
                print(f"[SemanticFloorDetector] Found {len(all_categories_found)} categories")
                if floor_like:
                    print(f"[SemanticFloorDetector] Floor-like: {floor_like}")
                else:
                    sample = sorted([c for c in all_categories_found if c])[:15]
                    print(f"[SemanticFloorDetector] Sample: {sample}")
            else:
                print("[SemanticFloorDetector] No categories found!")
            
            # compute floor heights
            for level_id, heights in floor_objects_by_level.items():
                if heights:
                    self._floor_heights[level_id] = sum(heights) / len(heights)
            
            self._sorted_floors = sorted(self._floor_heights.items(), key=lambda x: x[1])
            self._calculate_floor_ranges()
            
            if len(self._floor_heights) > 0:
                self._initialized = True
                print(f"[SemanticFloorDetector] Initialized with {len(self._floor_heights)} floors:")
                for floor_id, floor_y in self._sorted_floors:
                    range_min, range_max = self._floor_ranges.get(floor_id, (floor_y, floor_y))
                    print(f"  Floor {floor_id}: Y = {floor_y:.3f}m")
            else:
                print("[SemanticFloorDetector] Warning: No floor objects found")
                
        except Exception as e:
            import traceback
            print(f"[SemanticFloorDetector] Error: {e}")
            traceback.print_exc()
    
    def _calculate_floor_ranges(self):
        """Height range covered by each floor."""
        if len(self._sorted_floors) == 0:
            return
        
        for i, (floor_id, floor_y) in enumerate(self._sorted_floors):
            # lower bound: this floor's height
            range_min = floor_y
            
            # upper bound: the next floor up, or an arbitrary ceiling
            if i + 1 < len(self._sorted_floors):
                next_floor_y = self._sorted_floors[i + 1][1]
                range_max = next_floor_y
            else:
                range_max = floor_y + 10.0  # top floor, assume at most 10 m of headroom
            
            self._floor_ranges[floor_id] = (range_min, range_max)
    
    def is_initialized(self) -> bool:
        """Whether initialization succeeded."""
        return self._initialized
    
    def get_floor_count(self) -> int:
        """Number of floors."""
        return len(self._floor_heights)
    
    def get_floor_heights(self) -> Dict[int, float]:
        """Floor id to height mapping."""
        return dict(self._floor_heights)
    
    def get_floor_at_height(self, height: float) -> Optional[int]:
        """
        Floor containing a given height.
        
        Args:
            height: the Y coordinate
            
        Returns:
            A floor id, or None when undecidable
        """
        if not self._initialized:
            return None
        
        for floor_id, (range_min, range_max) in self._floor_ranges.items():
            if range_min <= height < range_max:
                return floor_id
        
        # below the lowest floor, return the lowest
        if len(self._sorted_floors) > 0:
            lowest_floor_id = self._sorted_floors[0][0]
            lowest_floor_y = self._sorted_floors[0][1]
            if height < lowest_floor_y:
                return lowest_floor_id
        
        return None
    
    def get_floor_at_position(self, position) -> Optional[int]:
        """
        Floor containing a 3D position.
        
        Args:
            position: [x, y, z]
            
        Returns:
            A floor id
        """
        if hasattr(position, '__len__') and len(position) >= 2:
            return self.get_floor_at_height(float(position[1]))
        return None
    
    def check_cross_floor(self, start_position, goal_position) -> Dict[str, Any]:
        """
        Check for a floor change using the semantic annotations.
        
        Args:
            start_position: [x, y, z]
            goal_position: [x, y, z]
            
        Returns:
            A dict describing the result
        """
        result = {
            "is_cross_floor": False,
            "start_floor": None,
            "goal_floor": None,
            "floor_difference": 0,
            "detection_method": "semantic",
            "initialized": self._initialized,
        }
        
        if not self._initialized:
            result["detection_method"] = "semantic_fallback_threshold"
            # fall back to the height threshold
            start_y = float(start_position[1]) if hasattr(start_position, '__len__') else 0
            goal_y = float(goal_position[1]) if hasattr(goal_position, '__len__') else 0
            height_diff = abs(goal_y - start_y)
            result["is_cross_floor"] = height_diff > 0.25  # default threshold
            result["height_diff"] = height_diff
            return result
        
        # floors of the start and the goal
        start_floor = self.get_floor_at_position(start_position)
        goal_floor = self.get_floor_at_position(goal_position)
        
        result["start_floor"] = start_floor
        result["goal_floor"] = goal_floor
        
        if start_floor is not None and goal_floor is not None:
            result["floor_difference"] = abs(goal_floor - start_floor)
            result["is_cross_floor"] = (start_floor != goal_floor)
        
        return result
    
    def check_cross_floor_by_path(self, sim, start_position, goal_position) -> Dict[str, Any]:
        """
        Check for a floor change along the shortest path, using the annotations.
        
        Args:
            sim: Habitat Simulator
            start_position: start
            goal_position: goal
            
        Returns:
            A dict with the full result
        """
        import habitat_sim
        
        result = {
            "is_cross_floor": False,
            "is_cross_floor_path": False,
            "path_found": False,
            "path_height_range": 0.0,
            "start_floor": None,
            "goal_floor": None,
            "floors_traversed": [],
            "detection_method": "semantic_path",
        }
        
        # 1. compare the floors of start and goal
        semantic_result = self.check_cross_floor(start_position, goal_position)
        result.update(semantic_result)
        
        # 2. check the floors the shortest path crosses
        try:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.array(start_position, dtype=np.float32)
            path.requested_end = np.array(goal_position, dtype=np.float32)
            found_path = sim.pathfinder.find_path(path)
            
            result["path_found"] = found_path
            
            if found_path and len(path.points) > 0:
                heights = [float(p[1]) for p in path.points]
                result["path_height_range"] = round(max(heights) - min(heights), 4)
                
                # every floor along the path
                floors_set = set()
                for point in path.points:
                    floor_id = self.get_floor_at_position(point)
                    if floor_id is not None:
                        floors_set.add(floor_id)
                
                result["floors_traversed"] = sorted(list(floors_set))
                
                # more than one floor means a crossing
                if len(floors_set) > 1:
                    result["is_cross_floor_path"] = True
                    result["is_cross_floor"] = True
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def analyze_cross_floor(
        self, 
        sim, 
        start_position, 
        goal_position, 
        trajectory_heights: List[float] = None,
        threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Combined cross-floor analysis, matching the interface used by the evaluator.
        
        Combines the semantic annotations with path analysis.
        
        Args:
            sim: the Habitat simulator
            start_position: episode start
            goal_position: goal
            trajectory_heights: walked heights, optional
            threshold: height threshold for the fallback, default 0.25 m
            
        Returns:
            A dict in the format the evaluator expects
        """
        # 1. semantic path analysis
        path_result = self.check_cross_floor_by_path(sim, start_position, goal_position)
        
        # 2. trajectory analysis, when available
        traj_is_cross = False
        traj_height_range = 0.0
        start_height = None
        end_height = None
        min_height = None
        max_height = None
        
        if trajectory_heights and len(trajectory_heights) > 0:
            heights = [float(h) for h in trajectory_heights]
            start_height = heights[0]
            end_height = heights[-1]
            min_height = min(heights)
            max_height = max(heights)
            traj_height_range = max_height - min_height
            
            # floors the trajectory crosses
            traj_floors = set()
            for h in heights:
                floor_id = self.get_floor_at_height(h)
                if floor_id is not None:
                    traj_floors.add(floor_id)
            
            if len(traj_floors) > 1:
                traj_is_cross = True
        
        # 3. combine
        is_cross_floor = path_result["is_cross_floor"] or traj_is_cross
        
        # assemble the explanation
        reasons = []
        if path_result.get("is_cross_floor_path"):
            floors = path_result.get("floors_traversed", [])
            reasons.append(f"path crosses floors: {floors}")
        if path_result.get("start_floor") != path_result.get("goal_floor"):
            reasons.append(f"start_floor={path_result.get('start_floor')} != goal_floor={path_result.get('goal_floor')}")
        if traj_is_cross:
            reasons.append(f"trajectory crosses multiple floors")
        
        return {
            "is_cross_floor": is_cross_floor,
            "cross_floor_reason": "; ".join(reasons) if reasons else None,
            "threshold_used": threshold,
            # semantic result
            "detection_method": "semantic" if self._initialized else "threshold_fallback",
            "start_floor": path_result.get("start_floor"),
            "goal_floor": path_result.get("goal_floor"),
            "floors_traversed": path_result.get("floors_traversed", []),
            # path result
            "is_cross_floor_path": path_result.get("is_cross_floor_path", False),
            "path_height_range": path_result.get("path_height_range", 0.0),
            "path_found": path_result.get("path_found", False),
            # trajectory result
            "is_cross_floor_trajectory": traj_is_cross,
            "trajectory_height_range": round(traj_height_range, 4),
            "height_range": round(traj_height_range, 4),  # kept for backward compatibility
            # heights
            "start_height": start_height,
            "end_height": end_height,
            "min_height": min_height,
            "max_height": max_height,
            # kept for backward compatibility
            "num_floor_clusters": len(path_result.get("floors_traversed", [])) or (2 if is_cross_floor else 1),
            "floor_heights": [self._floor_heights.get(f, 0) for f in path_result.get("floors_traversed", [])],
            "floor_levels": path_result.get("floors_traversed", []),
            "stair_segments": [],
            "total_stair_steps": 0,
            "num_stair_segments": 0,
            "max_single_climb": traj_height_range if is_cross_floor else 0.0,
            "max_single_descent": traj_height_range if is_cross_floor else 0.0,
        }


# ============================================================
# Convenience wrappers, drop-in replacements for the evaluator's own functions
# ============================================================

_detector_cache: Dict[int, SemanticFloorDetector] = {}


def get_floor_detector(sim) -> SemanticFloorDetector:
    """
    Get or build a cached detector.
    
    Args:
        sim: the simulator
        
    Returns:
        A SemanticFloorDetector
    """
    sim_id = id(sim)
    if sim_id not in _detector_cache:
        _detector_cache[sim_id] = SemanticFloorDetector(sim)
    return _detector_cache[sim_id]


def clear_detector_cache():
    """Clear the detector cache."""
    global _detector_cache
    _detector_cache = {}


def analyze_cross_floor(
    sim, 
    start_position, 
    goal_position, 
    trajectory_heights: List[float] = None,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Combined cross-floor analysis, replacing the evaluator's own function.
    
    Uses the semantic annotations when available, otherwise the height threshold.
    
    Args:
        sim: the Habitat simulator
        start_position: episode start
        goal_position: goal  
        trajectory_heights: walked heights
        threshold: height threshold for the fallback
        
    Returns:
        A dict with the full result
    """
    detector = get_floor_detector(sim)
    return detector.analyze_cross_floor(
        sim, start_position, goal_position, trajectory_heights, threshold
    )


if __name__ == "__main__":
    print("Semantic Floor Detector Module")
    print("Usage:")
    print("  from semantic_floor_detector import SemanticFloorDetector, analyze_cross_floor")
    print("")
    print("  # 1. the detector class")
    print("  detector = SemanticFloorDetector(sim)")
    print("  floor_id = detector.get_floor_at_position(position)")
    print("  result = detector.check_cross_floor(start_pos, goal_pos)")
    print("")
    print("  # 2. the convenience wrapper, a drop-in replacement")
    print("  result = analyze_cross_floor(sim, start_pos, goal_pos, trajectory_heights)")
