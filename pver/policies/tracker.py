from typing import List, Dict, Tuple, Optional
import re
import math


# ---------------------------------------------------------------------------
# Attribute Visibility Estimation
# ---------------------------------------------------------------------------
# Directional keywords → attribute faces a specific direction
_FRONT_KW = {"front", "face", "screen", "display", "dial", "facade", "chest"}
_BACK_KW  = {"back", "rear", "behind", "backside"}
# These are visible from any horizontal angle (we don't tilt the camera)
_NEUTRAL_KW = {"top", "upper", "lid", "cap", "crown",
               "bottom", "lower", "base", "foot", "feet", "side", "lateral"}
# Directionally ambiguous: might be on any face
_UNCERTAIN_KW = {"logo", "text", "label", "sticker", "tag", "emblem", "badge",
                 "print", "decal", "marking", "inscription"}


def _has_keyword(text: str, kw_set: set) -> bool:
    """Check if text contains any keyword as a whole word."""
    for kw in kw_set:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            return True
    return False


def estimate_visibility(attr_name: str, evidence_phrase: str, sector_idx: int) -> float:
    """
    Estimate how visible an attribute is from a given sector.

    Args:
        attr_name: e.g. "color", "front_logo", "desc_1"
        evidence_phrase: e.g. "red", "red circle on the front", full sentence
        sector_idx: virtual sector index 0-5 (0=front, 3=back)

    Returns:
        float in [0, 1]. 1.0 = fully visible, 0.0 = not visible.
    """
    # Combine name (underscores → spaces) and evidence for keyword search
    text = (attr_name.replace("_", " ") + " " + evidence_phrase).lower()

    # 1. Check directional keywords first
    has_front = _has_keyword(text, _FRONT_KW)
    has_back  = _has_keyword(text, _BACK_KW)

    if has_front and not has_back:
        attr_facing = 0  # degrees
    elif has_back and not has_front:
        attr_facing = 180
    elif has_front and has_back:
        return 0.5  # contradictory → treat as uncertain
    else:
        # No directional keyword
        if _has_keyword(text, _UNCERTAIN_KW):
            return 0.5  # logo/text with unknown facing
        return 1.0  # view-independent (color, material, shape, etc.)

    # Cosine decay: fully visible when facing camera, invisible from opposite side
    sector_angle = sector_idx * 60  # 0, 60, 120, 180, 240, 300
    diff = abs(sector_angle - attr_facing)
    if diff > 180:
        diff = 360 - diff
    return max(0.0, math.cos(math.radians(diff)))


class AttributeStateTracker:
    """
    Tracks attribute states across multiple views logic.
    Now supports observation confidence weighting.
    """
    def __init__(self, attr_names: List[str]):
        # history format: (view_id, state, confidence)
        self.history: Dict[str, List[Tuple[int, str, float]]] = {name: [] for name in attr_names}
        self.current_state: Dict[str, str] = {name: "Missing" for name in attr_names}
        # Confidence-weighted vote accumulator for convergence checks
        self.vote_counts: Dict[str, Dict[str, float]] = {
            name: {"Matched": 0.0, "Contradicted": 0.0}
            for name in attr_names
        }
        # Visibility-weighted evidence accumulators (for VisibilityWeightedFusion)
        self.evidence: Dict[str, Dict[str, float]] = {
            name: {"yes": 0.0, "no": 0.0}
            for name in attr_names
        }

    def update(self, view_id: int, attribute_states: Dict[str, str],
               observation_confidence: float = 1.0,
               visibility_scores: Optional[Dict[str, float]] = None):
        """
        Update tracker with new observation.

        Args:
            view_id: Sequential view number
            attribute_states: Dict mapping attr_name -> state ("Matched"/"Contradictory"/"Missing")
            observation_confidence: Detection confidence (0-1). GT bbox = 1.0, DINO = score
            visibility_scores: Optional dict mapping attr_name -> visibility (0-1).
                               Used for visibility-weighted evidence accumulation.
        """
        for name, state in attribute_states.items():
            norm_name = self._norm(name)
            target_key = None
            if name in self.history: target_key = name
            else:
                for k in self.history:
                    if self._norm(k) == norm_name:
                        target_key = k
                        break

            if target_key:
                self.history[target_key].append((view_id, state, observation_confidence))
                self._reconcile_state(target_key)
                # Update confidence-weighted vote accumulator (unchanged for backward compat)
                if state in ("Matched", "Contradictory"):
                    vote_key = "Matched" if state == "Matched" else "Contradicted"
                    self.vote_counts[target_key][vote_key] += observation_confidence

                # Update visibility-weighted evidence
                vis = (visibility_scores or {}).get(target_key, 1.0)
                if state == "Matched":
                    # Yes always counts at full detection confidence
                    self.evidence[target_key]["yes"] += observation_confidence
                elif state == "Contradictory":
                    # No weighted by visibility: low-vis "No" barely counts
                    self.evidence[target_key]["no"] += observation_confidence * vis

    def _reconcile_state(self, name: str):
        """
        Reconcile state using confidence-weighted majority voting.

        Logic:
        - If high-confidence "Contradictory" exists, mark as Contradictory
        - Otherwise, if any "Matched" exists with reasonable confidence, mark as Matched
        - Otherwise, mark as Missing
        """
        if not self.history[name]:
            self.current_state[name] = "Missing"
            return

        # Weighted voting
        matched_weight = 0.0
        contradictory_weight = 0.0
        missing_weight = 0.0

        # Low confidence threshold - observations below this are downweighted significantly
        LOW_CONF_THRESHOLD = 0.3

        for view_id, state, conf in self.history[name]:
            # Downweight low confidence observations
            effective_conf = conf if conf >= LOW_CONF_THRESHOLD else conf * 0.2

            if state == "Matched":
                matched_weight += effective_conf
            elif state == "Contradictory":
                contradictory_weight += effective_conf
            else:  # Missing
                missing_weight += effective_conf

        # Decision logic: prioritize Matched if it has strong support
        if contradictory_weight > matched_weight and contradictory_weight > missing_weight:
            self.current_state[name] = "Contradictory"
        elif matched_weight > 0.3:  # Consistent with LOW_CONF_THRESHOLD
            self.current_state[name] = "Matched"
        else:
            self.current_state[name] = "Missing"

    def _norm(self, s):
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def get_best_confidence(self, name: str) -> float:
        """
        Get the highest confidence among verified observations (Matched/Contradictory) for an attribute.

        Returns:
            float: Best confidence for verified observations, or 0.0 if no verified observations
        """
        if name not in self.history:
            return 0.0

        best_conf = 0.0
        for view_id, state, conf in self.history[name]:
            # Only consider verified states (Matched or Contradictory)
            if state in ("Matched", "Contradictory"):
                best_conf = max(best_conf, conf)

        return best_conf

    def is_converged(self, name: str, remaining_budget: int) -> bool:
        """
        True if attribute result can no longer change regardless of remaining views.

        Uses exact mathematical convergence with confidence-weighted vote counts.
        Each remaining step contributes at most 1.0 weight (max detection confidence).
        Tie → Contradicted (conservative), so Matched needs a strict lead.
        """
        vc = self.vote_counts.get(name, {})
        m = vc.get("Matched", 0.0)
        c = vc.get("Contradicted", 0.0)
        if m == 0.0 and c == 0.0:
            return False  # no decisive votes yet
        if m > c:
            # Contradicted needs at least (m - c) weight to tie; max per step = 1.0
            return remaining_budget * 1.0 < (m - c)
        else:
            # Matched needs strictly more than (c - m) weight to win; max per step = 1.0
            return remaining_budget * 1.0 <= (c - m)

    def all_converged(self, remaining_budget: int) -> bool:
        """True if every tracked attribute is converged."""
        return all(self.is_converged(name, remaining_budget) for name in self.vote_counts)

    def get_final_attr_decisions(self) -> Dict[str, str]:
        """
        Final per-attribute decision via raw majority vote.
        Matched > Contradicted → Matched
        Contradicted >= Matched (including tie) → Contradicted
        Both zero → Missing
        """
        result = {}
        for name, vc in self.vote_counts.items():
            m = vc.get("Matched", 0)
            c = vc.get("Contradicted", 0)
            if m == 0 and c == 0:
                result[name] = "Missing"
            elif m > c:
                result[name] = "Matched"
            else:
                result[name] = "Contradicted"  # tie → Contradicted (conservative)
        return result

    # ------------------------------------------------------------------
    # Visibility-weighted evidence methods
    # ------------------------------------------------------------------
    def get_evidence_status(self, yes_threshold: float = 0.7,
                            no_threshold: float = 0.7,
                            min_evidence: float = 0.5) -> Dict[str, str]:
        """
        Per-attribute status based on visibility-weighted evidence.

        Returns dict of attr_name -> one of:
          "verified"     : evidence_yes / total > yes_threshold
          "rejected"     : evidence_no  / total > no_threshold
          "uncertain"    : conflicting evidence
          "insufficient" : total evidence weight < min_evidence
        """
        result = {}
        for name, ev in self.evidence.items():
            total = ev["yes"] + ev["no"]
            if total < min_evidence:
                result[name] = "insufficient"
            elif ev["yes"] / total > yes_threshold:
                result[name] = "verified"
            elif ev["no"] / total > no_threshold:
                result[name] = "rejected"
            else:
                result[name] = "uncertain"
        return result

    def evidence_should_stop(self, yes_threshold: float = 0.7,
                             no_threshold: float = 0.7,
                             min_evidence: float = 0.5) -> bool:
        """
        Check if visibility-weighted evidence warrants early stopping.

        Stop when:
          - Any attribute is "rejected" (strong negative evidence)
          - All attributes are "verified" (strong positive evidence)
        """
        statuses = self.get_evidence_status(yes_threshold, no_threshold, min_evidence)
        if any(s == "rejected" for s in statuses.values()):
            return True
        if all(s == "verified" for s in statuses.values()):
            return True
        return False
