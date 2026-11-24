import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from src.detector import PPEDetector
from src.pose_analyzer import PoseSafetyAnalyzer

def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Calculate center (cx, cy) of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def is_inside(inner_center: Tuple[float, float], outer_bbox: List[int], y_min_ratio: float = 0.0, y_max_ratio: float = 1.0) -> bool:
    """Check if a center point falls within an outer bounding box (with optional vertical ratio window)."""
    cx, cy = inner_center
    x1, y1, x2, y2 = outer_bbox
    box_h = y2 - y1
    
    in_x = (x1 <= cx <= x2)
    in_y = (y1 + y_min_ratio * box_h <= cy <= y1 + y_max_ratio * box_h)
    return in_x and in_y

class PPETracker:
    """
    Tracks individuals in surveillance videos using YOLOv8 + ByteTrack
    and verifies personal protective equipment (Helmet, Vest/Sleeves, Gloves).
    Combines YOLO object detection with Pose Keypoint Analysis.
    """
    def __init__(
        self,
        tracking_weights: str = "models/yolov8n.pt",
        ppe_detector: PPEDetector = None,
        pose_weights: str = "models/yolov8n-pose.pt",
        conf: float = 0.35
    ):
        self.tracking_model = YOLO(tracking_weights)
        self.detector = ppe_detector if ppe_detector is not None else PPEDetector()
        self.pose_analyzer = PoseSafetyAnalyzer(pose_weights=pose_weights)
        self.conf = conf

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a single video frame:
        1. Tracks persons using ByteTrack.
        2. Detects PPE items (Helmet, Vest, Glasses).
        3. Runs pose-guided analysis on each tracked person (Sleeves, Gloves, Uniform).
        4. Fuses all checks into a consolidated status per person ID.
        """
        h, w = frame.shape[:2]
        
        # 1. Run ByteTrack on Persons
        track_results = self.tracking_model.track(
            source=frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],  # Track persons
            conf=self.conf,
            verbose=False
        )[0]

        tracked_persons = []
        if track_results.boxes is not None and track_results.boxes.id is not None:
            boxes = track_results.boxes.xyxy.cpu().numpy()
            track_ids = track_results.boxes.id.int().cpu().numpy()
            confs = track_results.boxes.conf.cpu().numpy()

            for box, tid, score in zip(boxes, track_ids, confs):
                xyxy = [int(v) for v in box]
                tracked_persons.append({
                    "track_id": int(tid),
                    "bbox": xyxy,
                    "conf": float(score),
                    "ppe_status": {
                        "helmet": False,
                        "vest": False,
                        "full_sleeves": False,
                        "gloves": False
                    }
                })

        # 2. Run Object Detector on Frame
        ppe_items = self.detector.detect(frame, conf=0.25, imgsz=1280)

        # 3. Spatial Association of Detected Items
        for item in ppe_items:
            cat = item["category"]
            c_pt = bbox_center(item["bbox"])

            for person in tracked_persons:
                p_box = person["bbox"]
                
                # Helmet: top 45% of height
                if cat == "helmet" and is_inside(c_pt, p_box, y_min_ratio=0.0, y_max_ratio=0.45):
                    person["ppe_status"]["helmet"] = True
                
                # Safety Vest: torso region (15% to 80%)
                elif cat == "vest" and is_inside(c_pt, p_box, y_min_ratio=0.15, y_max_ratio=0.80):
                    person["ppe_status"]["vest"] = True

                # Gloves: hands region (35% to 100%)
                elif cat == "gloves" and is_inside(c_pt, p_box, y_min_ratio=0.35, y_max_ratio=1.0):
                    person["ppe_status"]["gloves"] = True

        # 4. Pose-Guided Analysis on each Tracked Person
        for person in tracked_persons:
            x1, y1, x2, y2 = person["bbox"]
            px1, py1 = max(0, x1), max(0, y1)
            px2, py2 = min(w, x2), min(h, y2)
            
            if (px2 - px1) > 20 and (py2 - py1) > 40:
                crop = frame[py1:py2, px1:px2]
                pose_info = self.pose_analyzer.analyze_person(crop)
                
                # Full sleeves
                if pose_info["has_full_sleeves"]:
                    person["ppe_status"]["full_sleeves"] = True
                    person["ppe_status"]["vest"] = True  # Full sleeves boiler suit counts as safe uniform
                
                # Gloves
                if pose_info["has_gloves"]:
                    person["ppe_status"]["gloves"] = True

        return tracked_persons, ppe_items
