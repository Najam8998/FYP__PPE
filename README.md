# PPE Surveillance & Safety Audit System

An automated surveillance safety audit system built with **YOLOv8** and **ByteTrack** for tracking and auditing the 3 mandatory Personal Protective Equipment (**PPE**) categories:
1. **Hardhat / Helmet** (`H`)
2. **Safety Vest / Full Sleeves** (`V/S`)
3. **Work Gloves** (`GL`)

---

## Architecture & How It Works

* **ByteTrack Tracking**: Each worker is assigned a persistent tracking ID (e.g., `ID: #1`, `ID: #2`) across video frames.
* **Helmet Detection**: Uses fine-tuned YOLOv8 PPE detection with spatial matching in the head region.
* **Vest / Full Sleeves Check**: Uses YOLO detection for fluorescent vests and pose-guided forearm analysis (elbow $\rightarrow$ wrist) to verify coverage by full sleeves / boiler suit.
* **Glove Check**: Uses pose-guided wrist keypoints to analyze hand patches and verify work gloves vs. bare skin.
* **Compliance Standard**: A worker is labeled **`[SAFE]` (Green)** only if all 3 requirements are satisfied:
  $$\text{Compliant} = \text{Helmet} \land (\text{Vest} \lor \text{Full Sleeves}) \land \text{Gloves}$$
  Otherwise, the worker is flagged as **`[VIOLATION]` (Red)**.

---

## Project Structure

```
├── models/
│   ├── ppe_construction_yolov8.pt  # Fine-tuned PPE weights (Hardhat, Vest, Gloves)
│   ├── yolov8n-pose.pt             # Pose model for sleeve & hand keypoint inspection
│   └── yolov8n.pt                  # ByteTrack person detector & tracker
├── src/
│   ├── detector.py                 # Primary PPE object detector (HEIC & RGB)
│   ├── pose_analyzer.py           # Pose-guided sleeve, glove, and uniform analyzer
│   ├── tracker.py                  # ByteTrack tracking & 3-class status fusion
│   └── visualizer.py               # HUD overlay & 3-badge visualizer
├── PPE_PICS/                       # Photos (.HEIC, .jpg, .png)
├── PPE_VIDS/                       # Surveillance videos (.MOV, .mp4)
├── outputs/
│   ├── annotated_images/           # Output audited images
│   └── annotated_videos/           # Output tracked surveillance videos
├── detect_images.py                # Image folder detection CLI
└── track_video.py                  # Video tracking CLI
```

---

## Usage

### 1. Image Detection
Run PPE detection on images in `PPE_PICS`:
```bash
python detect_images.py --input PPE_PICS --output outputs/annotated_images --conf 0.25
```

### 2. Video Tracking with ByteTrack
Run tracking and safety audit on videos in `PPE_VIDS`:
```bash
python track_video.py --video PPE_VIDS/IMG_0475.MOV --output outputs/annotated_videos --max-frames 150
```
*(Omit `--max-frames` to process the entire video).*
