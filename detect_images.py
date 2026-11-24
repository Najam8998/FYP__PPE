import os
import cv2
import argparse
from src.detector import PPEDetector, load_image_bgr
from src.visualizer import draw_corner_rect, draw_hud, draw_ppe_item

def run_image_detection(
    input_dir: str = "PPE_PICS",
    output_dir: str = "outputs/annotated_images",
    conf: float = 0.25
):
    os.makedirs(output_dir, exist_ok=True)
    detector = PPEDetector(conf=conf)
    
    valid_exts = ('.heic', '.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    print("=" * 60)
    print("PPE SURVEILLANCE SYSTEM - IMAGE DETECTION")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Total Images     : {len(image_files)}")
    print("=" * 60)

    for idx, fname in enumerate(image_files, 1):
        in_path = os.path.join(input_dir, fname)
        print(f"\n[{idx}/{len(image_files)}] Processing: {fname}")
        
        # Load image (supporting HEIC format)
        img = load_image_bgr(in_path)
        detections = detector.detect(img, conf=conf)
        
        # Count items
        helmets = sum(1 for d in detections if d["category"] == "helmet")
        vests = sum(1 for d in detections if d["category"] == "vest")
        gloves = sum(1 for d in detections if d["category"] == "gloves")
        persons = sum(1 for d in detections if d["category"] == "person")
        
        print(f"   -> Detected: {len(detections)} total items (Helmets: {helmets}, Vests: {vests}, Gloves: {gloves}, Persons: {persons})")
        
        # Draw PPE bounding boxes
        annotated = img.copy()
        for item in detections:
            draw_ppe_item(annotated, item["label"], item["conf"], item["bbox"])
            
        # Draw Dashboard HUD
        stats = {
            "Total Objects": len(detections),
            "Helmets Detected": helmets,
            "Safety Vests": vests,
            "Gloves Detected": gloves,
            "Persons Detected": persons
        }
        draw_hud(annotated, "PPE AUDIT - IMAGE", stats)
        
        # Save output
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{base_name}_audit.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"   -> Saved: {out_path}")

    print("\n[SUCCESS] Image detection complete! Outputs saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect PPE in images")
    parser.add_argument("--input", default="PPE_PICS", help="Input directory")
    parser.add_argument("--output", default="outputs/annotated_images", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    run_image_detection(input_dir=args.input, output_dir=args.output, conf=args.conf)
