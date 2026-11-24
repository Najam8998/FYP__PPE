import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from ultralytics import YOLO

def is_skin_pixel_hsv(h: int, s: int, v: int) -> bool:
    """Determine whether an HSV pixel falls within common human skin tone ranges."""
    return (0 <= h <= 25 or 165 <= h <= 180) and (35 <= s <= 200) and (50 <= v <= 255)

class PoseSafetyAnalyzer:
    """
    Analyzes human pose keypoints to verify PPE compliance:
    - Full Sleeves (forearm skin exposure vs. clothing coverage)
    - Gloves (hand patch skin exposure vs. glove coverage)
    - Uniform / Vest (torso clothing & reflective stripe analysis)
    """
    def __init__(self, pose_weights: str = "models/yolov8n-pose.pt"):
        self.pose_model = YOLO(pose_weights)

    def analyze_person(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a cropped image of a tracked person.
        Returns:
            dict containing:
              - has_full_sleeves: bool
              - has_gloves: bool
              - has_uniform: bool
              - details: dict of confidence metrics
        """
        h_crop, w_crop = person_crop.shape[:2]
        if h_crop < 40 or w_crop < 20:
            return {
                "has_full_sleeves": False,
                "has_gloves": False,
                "has_uniform": False,
                "details": {}
            }

        # Run pose detection on the cropped person
        results = self.pose_model(person_crop, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            return {
                "has_full_sleeves": False,
                "has_gloves": False,
                "has_uniform": False,
                "details": {}
            }

        kpts = results.keypoints.xy.cpu().numpy()[0]
        confs = results.keypoints.conf.cpu().numpy()[0] if results.keypoints.conf is not None else [1.0] * 17
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

        # -------------------------------------------------------------
        # 1. Full Sleeves Check: Elbow to Wrist (kpt 7->9 and 8->10)
        # -------------------------------------------------------------
        arm_skin_ratios = []
        for side, (el_idx, wr_idx) in [("left", (7, 9)), ("right", (8, 10))]:
            if confs[el_idx] > 0.4 and confs[wr_idx] > 0.4:
                p_el, p_wr = kpts[el_idx], kpts[wr_idx]
                total_samples = 0
                skin_samples = 0
                for alpha in np.linspace(0.15, 0.85, 12):
                    pt = (p_el * (1 - alpha) + p_wr * alpha).astype(int)
                    for dx in [-2, 0, 2]:
                        for dy in [-2, 0, 2]:
                            x, y = pt[0] + dx, pt[1] + dy
                            if 0 <= x < w_crop and 0 <= y < h_crop:
                                total_samples += 1
                                h, s, v = hsv[y, x]
                                if is_skin_pixel_hsv(int(h), int(s), int(v)):
                                    skin_samples += 1
                if total_samples > 0:
                    arm_skin_ratios.append(skin_samples / total_samples)

        # If forearms have < 25% skin pixels, they are covered by full sleeves
        has_full_sleeves = False
        if arm_skin_ratios:
            avg_arm_skin = sum(arm_skin_ratios) / len(arm_skin_ratios)
            has_full_sleeves = (avg_arm_skin < 0.25)
        else:
            # Fallback if arm keypoints obscured
            has_full_sleeves = True

        # -------------------------------------------------------------
        # 2. Glove Check: Hand / Wrist Patches (kpt 9 and 10)
        # -------------------------------------------------------------
        hand_skin_ratios = []
        for wr_idx in [9, 10]:
            if confs[wr_idx] > 0.4:
                wx, wy = int(kpts[wr_idx][0]), int(kpts[wr_idx][1])
                patch_r = max(8, int(min(w_crop, h_crop) * 0.05))
                y_min, y_max = max(0, wy - patch_r), min(h_crop, wy + patch_r)
                x_min, x_max = max(0, wx - patch_r), min(w_crop, wx + patch_r)
                patch = hsv[y_min:y_max, x_min:x_max]

                if patch.size > 0:
                    skin_px = 0
                    total_px = 0
                    for row in patch:
                        for pix in row:
                            total_px += 1
                            if is_skin_pixel_hsv(int(pix[0]), int(pix[1]), int(pix[2])):
                                skin_px += 1
                    if total_px > 0:
                        hand_skin_ratios.append(skin_px / total_px)

        # Work gloves (black, leather, colored rubber) mask bare skin (< 20% skin pixels)
        has_gloves = False
        if hand_skin_ratios:
            avg_hand_skin = sum(hand_skin_ratios) / len(hand_skin_ratios)
            has_gloves = (avg_hand_skin < 0.20)
        else:
            has_gloves = False

        # -------------------------------------------------------------
        # 3. Workwear Uniform / Reflective Tape Check (Torso)
        # -------------------------------------------------------------
        # Check torso area between shoulders (5, 6) and hips (11, 12)
        has_uniform = False
        if confs[5] > 0.4 and confs[6] > 0.4:
            y_sh = int((kpts[5][1] + kpts[6][1]) / 2.0)
            y_hip = int((kpts[11][1] + kpts[12][1]) / 2.0) if (confs[11] > 0.3 and confs[12] > 0.3) else int(y_sh + h_crop * 0.4)
            torso_patch = person_crop[max(0, y_sh):min(h_crop, y_hip), :]
            if torso_patch.size > 0:
                gray_torso = cv2.cvtColor(torso_patch, cv2.COLOR_BGR2GRAY)
                # Look for high-contrast reflective stripes / uniform edges
                edges = cv2.Canny(gray_torso, 50, 150)
                edge_density = np.count_nonzero(edges) / max(1, edges.size)
                # Uniform / clothing is present if edge density / fabric is present
                has_uniform = (edge_density > 0.03)

        return {
            "has_full_sleeves": has_full_sleeves,
            "has_gloves": has_gloves,
            "has_uniform": has_uniform,
            "details": {
                "arm_skin_ratios": arm_skin_ratios,
                "hand_skin_ratios": hand_skin_ratios
            }
        }
