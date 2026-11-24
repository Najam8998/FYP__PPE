import os
import cv2
import argparse
from tqdm import tqdm
from src.tracker import PPETracker
from src.visualizer import draw_person_track, draw_ppe_item, draw_hud

def process_video(
    video_path: str,
    output_dir: str = "outputs/annotated_videos",
    max_frames: int = None,
    conf: float = 0.30
):
    os.makedirs(output_dir, exist_ok=True)
    tracker = PPETracker(conf=conf)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames > 0 and max_frames < total_frames:
        total_frames = max_frames
    else:
        max_frames = None
        
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    print("=" * 60)
    print("PPE SURVEILLANCE & BYTETRACK TRACKING (WITH POSE-GUIDED PPE)")
    print(f"Source Video     : {video_path}")
    print(f"Resolution       : {width}x{height} @ {fps:.1f} FPS")
    print(f"Total Frames     : {total_frames}")
    print(f"Output File      : {out_video_path}")
    print("=" * 60)
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {base_name}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
            
        # Run ByteTrack & PPE association
        tracked_persons, ppe_items = tracker.process_frame(frame, frame_idx=frame_idx)
        
        # Draw PPE items first (in background)
        for item in ppe_items:
            if item["category"] in ["helmet", "vest", "gloves"]:
                draw_ppe_item(frame, item["label"], item["conf"], item["bbox"])
                
        # Draw Tracked Persons with ByteTrack ID and compliance badges
        active_persons = len(tracked_persons)
        helmets_count = 0
        vests_count = 0
        gloves_count = 0
        compliant_persons = 0
        
        for person in tracked_persons:
            status = person["ppe_status"]
            has_h = status.get("helmet", False)
            has_v = status.get("vest", False) or status.get("full_sleeves", False)
            has_gl = status.get("gloves", False)
            
            if has_h:
                helmets_count += 1
            if has_v:
                vests_count += 1
            if has_gl:
                gloves_count += 1
                
            # Mandatory safety compliance: Helmet AND (Vest or Full Sleeves) AND Gloves
            if has_h and has_v and has_gl:
                compliant_persons += 1
                
            draw_person_track(frame, person["track_id"], person["bbox"], status)
            
        violations = active_persons - compliant_persons
        compliance_pct = f"{(compliant_persons / active_persons * 100):.1f}%" if active_persons > 0 else "N/A"
        
        # HUD overlay
        stats = {
            "Frame": f"{frame_idx} / {total_frames}",
            "Tracked Persons": active_persons,
            "Helmets Detected": f"{helmets_count}/{active_persons}",
            "Sleeves/Vest OK": f"{vests_count}/{active_persons}",
            "Gloves Detected": f"{gloves_count}/{active_persons}",
            "Compliant (Safe)": compliant_persons,
            "Violations (Risk)": violations,
            "Compliance Rate": compliance_pct
        }
        draw_hud(frame, "SURVEILLANCE SAFETY AUDIT", stats)
        
        writer.write(frame)
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    writer.release()
    print(f"\n[DONE] Saved tracked video to: {out_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Track PPE in video using ByteTrack")
    parser.add_argument("--video", default="PPE_VIDS/IMG_0475.MOV", help="Path to video file")
    parser.add_argument("--output", default="outputs/annotated_videos", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to process for quick testing (default: 60)")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    args = parser.parse_args()
    
    process_video(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        conf=args.conf
    )

if __name__ == "__main__":
    main()
