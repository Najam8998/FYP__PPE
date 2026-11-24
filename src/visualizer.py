import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Color Palette (BGR format)
COLOR_SAFE = (72, 201, 72)      # Green
COLOR_WARNING = (0, 165, 255)   # Amber
COLOR_DANGER = (50, 50, 230)    # Red
COLOR_CYAN = (238, 206, 0)      # Cyan
COLOR_MAGENTA = (180, 50, 200)  # Magenta
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 20, 20)

CLASS_COLORS = {
    "hardhat": (0, 215, 255),       # Gold/Yellow
    "helmet": (0, 215, 255),
    "safety vest": (0, 165, 255),   # Bright Orange
    "vest": (0, 165, 255),
    "gloves": (200, 100, 200),      # Purple/Magenta
    "no-hardhat": (40, 40, 220),
    "no_helmet": (40, 40, 220),
    "no-safety vest": (40, 40, 220),
    "no_glove": (40, 40, 220)
}

def draw_corner_rect(img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2, corner_len: int = 15):
    """Draw a modern rounded/corner-style bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

def draw_hud(img: np.ndarray, title: str, stats: Dict[str, Any]):
    """Draw a surveillance dashboard overlay on top-left of frame."""
    hud_w = 350
    hud_h = 42 + len(stats) * 26
    
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    
    cv2.rectangle(img, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, title, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (25, 46), (15 + hud_w - 10, 46), (120, 120, 120), 1)
    
    y = 70
    for key, val in stats.items():
        cv2.putText(img, f"{key}:", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        
        val_str = str(val)
        val_color = COLOR_WHITE
        if "Violation" in key and isinstance(val, int) and val > 0:
            val_color = COLOR_DANGER
        elif "Compliance" in key:
            val_color = COLOR_SAFE
            
        cv2.putText(img, val_str, (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, val_color, 2, cv2.LINE_AA)
        y += 26

def draw_person_track(img: np.ndarray, track_id: int, bbox: Tuple[int, int, int, int], ppe_status: Dict[str, bool]):
    """Draw a tracked person with ByteTrack ID and the 3 mandatory PPE badges (Helmet, Vest/Sleeves, Gloves)."""
    x1, y1, x2, y2 = bbox
    
    has_helmet = ppe_status.get("helmet", False)
    has_vest = ppe_status.get("vest", False) or ppe_status.get("full_sleeves", False)
    has_gloves = ppe_status.get("gloves", False)
    
    # 3 Mandatory PPE items: Helmet + Vest/Sleeves + Gloves
    is_compliant = has_helmet and has_vest and has_gloves
    box_color = COLOR_SAFE if is_compliant else COLOR_DANGER
    
    draw_corner_rect(img, (x1, y1, x2, y2), box_color, thickness=2)
    
    status_label = "SAFE" if is_compliant else "VIOLATION"
    header_text = f"ID: #{track_id} [{status_label}]"
    
    (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(img, (x1, max(0, y1 - 24)), (x1 + tw + 10, y1), box_color, -1)
    cv2.putText(img, header_text, (x1 + 5, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Badges for the 3 target classes: Helmet, Vest/Sleeves, Gloves
    badges = [
        f"H:{'OK' if has_helmet else 'NO'}",
        f"V/S:{'OK' if has_vest else 'NO'}",
        f"GL:{'OK' if has_gloves else 'NO'}"
    ]
    badge_str = " | ".join(badges)
    
    (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bg_y = y1 + 18 if (y1 + 25 < y2) else y2 + 18
    cv2.rectangle(img, (x1, bg_y - 14), (x1 + bw + 8, bg_y + 4), COLOR_DARK_BG, -1)
    cv2.putText(img, badge_str, (x1 + 4, bg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

def draw_ppe_item(img: np.ndarray, label: str, conf: float, bbox: Tuple[int, int, int, int]):
    """Draw a detected PPE item bounding box."""
    x1, y1, x2, y2 = bbox
    color = CLASS_COLORS.get(label.lower(), COLOR_CYAN)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    tag = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, tag, (x1 + 3, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
