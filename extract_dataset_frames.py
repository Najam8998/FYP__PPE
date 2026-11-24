import os
import cv2
from PIL import Image
import pillow_heif

# Register HEIC opener
pillow_heif.register_heif_opener()

def extract_frames(
    vids_dir: str = "PPE_VIDS",
    pics_dir: str = "PPE_PICS",
    output_dir: str = "dataset_raw/images",
    frames_per_second_sample: float = 2.5
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print("DATASET EXTRACTION PIPELINE FOR ANNOTATION")
    print(f"Video Source Folder : {vids_dir}")
    print(f"Photo Source Folder : {pics_dir}")
    print(f"Target Output Folder: {output_dir}")
    print("=" * 65)
    
    total_extracted = 0

    # 1. Extract from all videos in PPE_VIDS
    video_files = sorted([f for f in os.listdir(vids_dir) if f.lower().endswith(('.mov', '.mp4', '.avi'))])
    
    for vid_name in video_files:
        vid_path = os.path.join(vids_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        
        if not cap.isOpened():
            print(f"[WARNING] Skipping unreadable video: {vid_name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base_name = os.path.splitext(vid_name)[0]
        
        # Calculate frame step based on desired frames per second
        step = max(1, int(fps / frames_per_second_sample))
        
        print(f"\nProcessing {vid_name} ({total_f} frames, {fps:.1f} FPS, sampling every {step} frames)...")
        
        frame_idx = 0
        vid_extracted = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                out_filename = f"{base_name}_frame_{frame_idx:04d}.jpg"
                out_path = os.path.join(output_dir, out_filename)
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                vid_extracted += 1
                total_extracted += 1
                
            frame_idx += 1
            
        cap.release()
        print(f"   -> Extracted {vid_extracted} frames from {vid_name}")

    # 2. Convert and include photos from PPE_PICS
    if os.path.exists(pics_dir):
        pic_files = sorted([f for f in os.listdir(pics_dir) if f.lower().endswith(('.heic', '.jpg', '.jpeg', '.png'))])
        print(f"\nProcessing photos from {pics_dir} ({len(pic_files)} photos)...")
        
        for pic_name in pic_files:
            pic_path = os.path.join(pics_dir, pic_name)
            base_name = os.path.splitext(pic_name)[0]
            out_filename = f"PHOTO_{base_name}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            
            try:
                pil_img = Image.open(pic_path)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(out_path, "JPEG", quality=95)
                total_extracted += 1
                print(f"   -> Converted & saved {out_filename}")
            except Exception as e:
                print(f"   [ERROR] Failed to convert {pic_name}: {e}")

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Extraction Complete! Total {total_extracted} high-quality images ready for labeling.")
    print(f"Location: {os.path.abspath(output_dir)}")
    print("=" * 65)

if __name__ == "__main__":
    extract_frames()
