import os
import cv2
import numpy as np
from PIL import Image
import pillow_heif
from typing import List, Dict, Any, Union
from ultralytics import YOLO

# Register HEIC opener
pillow_heif.register_heif_opener()

def load_image_bgr(path: str) -> np.ndarray:
    """Load an image (supporting HEIC, JPG, PNG) and return an OpenCV BGR numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
        
    pil_img = Image.open(path)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    rgb_arr = np.array(pil_img)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

class PPEDetector:
    """
    Detector for Personal Protective Equipment:
    Target Classes: Helmet/Hardhat, Safety Vest/Workwear, Gloves.
    """
    def __init__(
        self,
        ppe_weights: str = "models/ppe_construction_yolov8.pt",
        conf: float = 0.25,
        device: str = "auto"
    ):
        self.conf = conf
        self.ppe_model = YOLO(ppe_weights)
        
        # Mapping model raw classes to standard categories
        self.category_map = {
            "hardhat": "helmet",
            "helmet": "helmet",
            "safety vest": "vest",
            "vest": "vest",
            "gloves": "gloves",
            "person": "person"
        }

    def detect(self, img_input: Union[str, np.ndarray], conf: float = None, imgsz: int = 1280) -> List[Dict[str, Any]]:
        """
        Run inference on an image (path or BGR array) and return structured detections.
        """
        if isinstance(img_input, str):
            bgr_img = load_image_bgr(img_input)
        else:
            bgr_img = img_input

        conf_thresh = conf if conf is not None else self.conf
        detections = []

        ppe_res = self.ppe_model.predict(bgr_img, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
        for box in ppe_res.boxes:
            cls_id = int(box.cls[0])
            raw_label = self.ppe_model.names[cls_id]
            score = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            
            category = self.category_map.get(raw_label.lower(), raw_label.lower())
            # Only retain relevant classes (helmet, vest, gloves, person)
            if category in ["helmet", "vest", "gloves", "person"]:
                detections.append({
                    "label": raw_label,
                    "category": category,
                    "conf": score,
                    "bbox": xyxy
                })

        return detections
