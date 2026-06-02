"""
End-to-end Trained Policy for SFT / SFT+GRPO / SFT+GSPO models.

The trained model makes ALL decisions itself in a single VL call per step:
  - Navigation: which sectors are navigable
  - Verification: does the object match the target description
  - Action: STOP or MOVE <sector>

No external tracker, fusion, or NBV modules needed.
"""

import os
import io
import re
import json
import base64
from PIL import Image

from pver.policies.server_client import ServerClient

# Must match training data format (prepare_sft_data.py)
SECTOR_NAMES = ["front", "front-left", "back-left", "back", "back-right", "front-right"]

SYSTEM_PROMPT = (
    "You are an embodied agent navigating around an object to verify whether it matches "
    "a target description. Each step you see two images: a full scene view and a close-up "
    "of the detected object. You must verify object attributes and decide your next action."
)

# NOTE: No <image><image> prefix — images are sent as content items by the server.
# During SFT training, <image> tags were placeholders replaced by the processor.
# At inference, images are injected via the multi-image endpoint instead.
USER_TEMPLATE = """You are an embodied agent verifying whether a detected object matches a target description.

Target: "{query_description}"
Category (must match): {query_category}
Current sector: {sector_name} ({angle}°)
Visited sectors: [{visited_list}]
Remaining budget: {remaining} steps
Available sectors: [{available_sectors}]

From the scene image (Image 1) and the object close-up (Image 2):
1. Does this object match the target description? Check each attribute.
2. Your action: STOP (if confident) or MOVE <sector> (if need more views)"""


def parse_answer_block(text):
    """Extract structured fields from <answer>...</answer> block.
    Falls back to fuzzy extraction from free-form text if no <answer> tag."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        block = match.group(1).strip()
        result = {}
        for line in block.split("\n"):
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip().lower()] = val.strip()
        return result

    # Fuzzy fallback for unstructured outputs (e.g., base model without SFT)
    result = {}
    text_lower = text.lower()

    # Extract action: look for MOVE <direction> or STOP
    move_match = re.search(r'\bMOVE\s+(front-left|front-right|back-left|back-right|back|front)\b',
                           text, re.IGNORECASE)
    if move_match:
        result["action"] = f"MOVE {move_match.group(1).lower()}"
    elif re.search(r'\bSTOP\b', text, re.IGNORECASE):
        result["action"] = "STOP"

    # Extract verification: look for explicit yes/no judgment
    if re.search(r'\b(does not match|doesn\'t match|not a match|mismatch)\b', text_lower):
        result["verification"] = "No"
    elif re.search(r'\b(matches the target|fully matches|exact match|confirmed match)\b', text_lower):
        result["verification"] = "Yes"
    # If action is MOVE, verification is implicitly Unsure
    elif result.get("action", "").startswith("MOVE"):
        result["verification"] = "Unsure"

    if result:
        result["_fuzzy_parsed"] = True
    return result


class TrainedPolicy:
    def __init__(self, cfg, client: ServerClient):
        self.cfg = cfg
        self.client = client
        self.max_steps = cfg.env.max_steps
        self.sectors = cfg.env.get("sectors", 6)

        # Load category cache for coarse class (used in DINO mode)
        self.category_cache = {}
        cache_dir = None
        if self.cfg.dataset.get("category_cache_path"):
            try:
                with open(self.cfg.dataset.category_cache_path, 'r', encoding='utf-8') as f:
                    self.category_cache = json.load(f)
                print(f"[TrainedPolicy] Loaded {len(self.category_cache)} categories from cache.")
                cache_dir = os.path.dirname(self.cfg.dataset.category_cache_path)
            except Exception as e:
                print(f"[TrainedPolicy] Failed to load category cache: {e}")

    def reset(self, obs):
        """Initialize for new episode."""
        meta = obs.get("meta", {})
        self.current_abs_idx = obs.get("current_sector", 0)
        self.visited_abs_indices = {self.current_abs_idx}
        self._failed_abs_sectors = set()
        self._last_target_abs_idx = None
        self.visited_sector_visibility = {}
        self.visited_views = set()
        self.step_history = []

        # Get query info (prefer JSONL index fields)
        self.object_id = str(meta.get("query_object_id") or meta.get("object_id") or "unknown")
        query_descs = meta.get("query_descriptions") or meta.get("text_pos") or [""] * 3
        self.query_descs = query_descs
        # Use all descriptions (complementary attributes across 3 descriptions)
        self.query_desc = "; ".join(d for d in query_descs if d) if query_descs else ""
        self.query_category = str(meta.get("query_object_category")
                                  or meta.get("target_object_category")
                                  or "object")

        # Coarse class for DINO
        if self.object_id in self.category_cache:
            self.target_class = self.category_cache[self.object_id].get("pred_coarse", "object")
        else:
            self.target_class = self.query_category

        self.current_episode = meta

        # For viz compatibility
        self.extraction_dialogue = {"mode": "trained_e2e", "descriptions": query_descs}
        self.coarse_dialogue = {
            "mode": "cache" if self.object_id in self.category_cache else "from_index",
            "result": self.target_class,
        }
        self.tracker = None  # No tracker needed

    def act(self, obs):
        """Single model call per step → parse decision."""
        # Track navigation failures
        if obs.get("navigation_failed") and self._last_target_abs_idx is not None:
            self._failed_abs_sectors.add(self._last_target_abs_idx)
            self.visited_abs_indices.add(self._last_target_abs_idx)
        self._last_target_abs_idx = None

        # Sync sector from env
        env_sector = obs.get("current_sector", -1)
        if env_sector >= 0:
            self.current_abs_idx = env_sector
            self.visited_abs_indices.add(env_sector)

        step_count = obs.get("step_count", 0)
        remaining = self.max_steps - step_count - 1
        is_final = remaining <= 0

        # 1. Get scene image and crop
        img_path = obs.get("rgb_path", "")
        crop_img, detection_confidence = self._get_crop(obs, img_path)

        # Record visibility
        meta = obs.get("meta", {})
        captures = meta.get("captures", [])
        base_name = os.path.basename(img_path)
        is_visible = True
        for cap in captures:
            c_path = cap.get("rgb_filename") or cap.get("rgb", "")
            if os.path.basename(c_path) == base_name:
                is_visible = cap.get("mask_meets_threshold", True)
                break
        self.visited_sector_visibility[self.current_abs_idx] = is_visible

        # 2. Build prompt — all directions are RELATIVE to current position
        # Agent always perceives its current position as "front" (0°)
        sector_name = "front"
        angle = 0

        # Visited sectors in relative terms from current position
        visited_names = []
        for abs_idx in sorted(self.visited_abs_indices):
            rel_offset = (abs_idx - self.current_abs_idx) % 6
            visited_names.append(SECTOR_NAMES[rel_offset])
        visited_list = ", ".join(visited_names)

        # Available sectors = ALL unvisited in relative terms
        excluded_rel = set()
        for abs_idx in self.visited_abs_indices | self._failed_abs_sectors:
            rel_offset = (abs_idx - self.current_abs_idx) % 6
            excluded_rel.add(rel_offset)
        available = [SECTOR_NAMES[i] for i in range(6) if i not in excluded_rel]
        available_str = ", ".join(available) if available else "none"

        user_text = USER_TEMPLATE.format(
            query_description=self.query_desc,
            query_category=self.query_category,
            sector_name=sector_name,
            angle=angle,
            visited_list=visited_list,
            remaining=remaining,
            available_sectors=available_str,
        )

        # Add visibility warning
        if obs.get("navigation_failed"):
            failed_dir = obs.get("failed_direction", "unknown")
            user_text += f"\nWarning: Navigation to {failed_dir} failed (unreachable)."
        elif not is_visible:
            user_text += "\nWarning: Target object is not clearly visible from this angle."

        # Add step history for temporal context
        # Directions are converted to be relative to CURRENT position
        if self.step_history:
            history_lines = []
            for h in self.step_history:
                rel_pos = SECTOR_NAMES[(h['abs_sector'] - self.current_abs_idx) % 6]
                if h.get('abs_target') is not None:
                    rel_target = SECTOR_NAMES[(h['abs_target'] - self.current_abs_idx) % 6]
                    action_text = f"MOVE {rel_target}"
                else:
                    action_text = h['action']
                history_lines.append(
                    f"- Step {h['step']} ({rel_pos}): "
                    f"Verification={h['verification']}, Action={action_text}"
                )
            user_text += "\n\nPrevious observations:\n" + "\n".join(history_lines)

        # 3. Call model with dual images (scene + crop) + system prompt
        result = self.client.call_qwen_vl_multi(
            images=[img_path, crop_img],
            prompt=user_text,
            system=SYSTEM_PROMPT,
        )
        response_text = result.get("text", "")

        # 4. Parse response
        answer = parse_answer_block(response_text)
        verification = answer.get("verification", "Unsure").strip()
        action_str = answer.get("action", "STOP").strip()

        # Normalize verification
        if verification.lower().startswith("yes"):
            verification = "Yes"
        elif verification.lower().startswith("no"):
            verification = "No"
        else:
            verification = "Unsure"

        # 5. Determine decision
        # Model uses relative directions (trained with relative naming):
        #   "MOVE front-left" means move 60° left of current position
        # This IS the nav_rel the env expects — no conversion needed.
        if action_str.upper().startswith("STOP") or is_final:
            decision = "No" if verification == "Unsure" else verification
            action = {"decision": decision, "nav_rel": "front"}
        elif action_str.upper().startswith("MOVE"):
            rel_name = self._parse_move_direction(action_str)
            # nav_rel = model's relative direction directly
            action = {"decision": "Unsure", "nav_rel": rel_name}
            # Track absolute target sector for failure tracking
            rel_offset = SECTOR_NAMES.index(rel_name) if rel_name in SECTOR_NAMES else 0
            self._last_target_abs_idx = (self.current_abs_idx + rel_offset) % 6
        else:
            # Unparseable → fallback to STOP No
            action = {"decision": "No", "nav_rel": "front"}

        self.visited_views.add(img_path)

        # Record step history with absolute sectors for correct relative rendering
        self.step_history.append({
            "step": step_count + 1,
            "abs_sector": self.current_abs_idx,
            "abs_target": self._last_target_abs_idx,  # None for STOP
            "verification": verification,
            "action": action_str,  # raw for debug
        })

        # Forced prediction at this step
        if action["decision"] == "Unsure":
            action["_step_prediction"] = "No" if verification != "Yes" else "Yes"
        else:
            action["_step_prediction"] = action["decision"]

        # --- Debug info for visualization ---
        action["_coarse_cat"] = self.target_class
        action["_coarse_source"] = "Cache" if self.object_id in self.category_cache else "Index"
        action["_detection_confidence"] = detection_confidence
        action["_current_abs_sector"] = sector_name
        action["_visited_abs_sectors"] = visited_names
        action["_sector_debug"] = {
            "env_returned_sector": obs.get("current_sector", -1),
            "policy_current_abs_idx": self.current_abs_idx,
            "policy_visited_abs_indices": sorted(list(self.visited_abs_indices)),
            "model_target_sector": action_str if action_str.upper().startswith("MOVE") else None,
            "env_nav_rel": action.get("nav_rel"),
        }

        # Raw model output
        action["_trained_raw_response"] = response_text
        action["_trained_parsed"] = answer

        # Per-attribute results (for viz compatibility, single e2e entry)
        action["_per_attribute_results"] = [{
            "name": "e2e_verification",
            "expected": self.query_desc[:80],
            "observed": verification,
            "accumulated_status": verification,
        }]

        action["_step_dialogues"] = [{
            "type": "trained_e2e",
            "prompt": user_text,
            "response": response_text,
            "parsed_verification": verification,
            "parsed_action": action_str,
        }]

        if len(self.visited_views) == 1:
            action["_extraction_dialogue"] = self.extraction_dialogue
            action["_coarse_dialogue"] = self.coarse_dialogue

        action["_nbv_debug"] = {
            "mode": "trained_e2e",
            "model_action": action_str,
        }
        action["_fusion_debug"] = {
            "mode": "trained_e2e",
            "verification": verification,
            "action": action_str,
        }
        action["_reason"] = f"trained: {action_str}"

        # Encode crop for HTML viz
        if crop_img and isinstance(crop_img, Image.Image):
            try:
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                action["_debug_crop"] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                pass

        return action

    # ------ helpers ------

    def _get_crop(self, obs, img_path):
        """Get cropped image using GT bbox or DINO."""
        bbox_mode = self.cfg.method.get("bbox_mode", "gt")

        if bbox_mode == "gt":
            meta = obs.get("meta", {})
            captures = meta.get("captures", [])
            base_name = os.path.basename(img_path)

            found_cap = None
            for cap in captures:
                c_path = cap.get("rgb_filename") or cap.get("rgb", "")
                if os.path.basename(c_path) == base_name:
                    found_cap = cap
                    break

            if found_cap and found_cap.get("mask_bbox_xyxy") is not None:
                box_result = {"results": [{"score": 1.0, "box": found_cap["mask_bbox_xyxy"]}]}
                confidence = 1.0
            else:
                box_result = {"results": []}
                confidence = 0.1
        else:
            # DINO mode
            box_result = self.client.call_gdino(img_path, self.target_class)
            results = box_result.get("results", [])
            confidence = max(r.get("score", 0.5) for r in results) if results else 0.1

        return self._crop_image(img_path, box_result), confidence

    def _crop_image(self, img_path, gdino_res):
        """Crop image, matching mllm_policy._crop_image."""
        try:
            img = Image.open(img_path).convert("RGB")
            results = gdino_res.get("results", [])
            if results:
                best = max(results, key=lambda x: x["score"])
                box = best["box"]
                w, h = img.size
                pad = 3
                x1 = max(0, min(w, box[0] - pad))
                y1 = max(0, min(h, box[1] - pad))
                x2 = max(0, min(w, box[2] + pad))
                y2 = max(0, min(h, box[3] + pad))
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    cw, ch = crop.size
                    if min(cw, ch) < 512:
                        s = 512.0 / min(cw, ch)
                        crop = crop.resize((int(cw * s + 0.5), int(ch * s + 0.5)), Image.BICUBIC)
                    return crop
            w, h = img.size
            if min(w, h) < 512:
                s = 512.0 / min(w, h)
                img = img.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BICUBIC)
            return img
        except Exception:
            return Image.new("RGB", (224, 224))

    def _parse_move_direction(self, action_str):
        """Parse direction from 'MOVE <direction>' string."""
        parts = action_str.strip().split(None, 1)
        if len(parts) >= 2:
            direction = parts[1].strip().lower()
            # Exact match first
            if direction in SECTOR_NAMES:
                return direction
            # Substring match — longest names first to avoid "front" matching "front-right"
            for name in sorted(SECTOR_NAMES, key=len, reverse=True):
                if name in direction:
                    return name
            # Fuzzy fallback
            mappings = {"left": "front-left", "right": "front-right",
                        "fl": "front-left", "fr": "front-right",
                        "bl": "back-left", "br": "back-right"}
            if direction in mappings:
                return mappings[direction]
        return "front"
