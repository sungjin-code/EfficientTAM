import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def create_mini_stage1(data_root, example_img_path):
    print("\n[1/2] Preparing Mini SA-1B (Stage 1) Dataset...")
    stage1_dir = data_root / "mini_SA1B"
    img_dir = stage1_dir / "images"
    mask_dir = stage1_dir / "masks"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    if not example_img_path.exists():
        print(f"Error: Example image not found at {example_img_path}")
        return
        
    # Copy image
    img_name = example_img_path.name
    shutil.copy2(example_img_path, img_dir / img_name)
    
    # Create dummy mask for the image
    img = cv2.imread(str(example_img_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a simple white rectangle in the center as a dummy mask
    center_x, center_y = w // 2, h // 2
    cv2.rectangle(mask, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), 255, -1)
    
    mask_name = f"{example_img_path.stem}_1.png"
    cv2.imwrite(str(mask_dir / mask_name), mask)
    print(f"Stage 1 Mini Dataset created at {stage1_dir}")

def create_mini_stage2(data_root, example_video_path):
    print("\n[2/2] Preparing Mini DAVIS (Stage 2) Dataset...")
    stage2_dir = data_root / "mini_DAVIS"
    img_dir = stage2_dir / "JPEGImages" / "test_video"
    ann_dir = stage2_dir / "Annotations" / "test_video"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    if not example_video_path.exists():
        print(f"Error: Example video not found at {example_video_path}")
        return
        
    cap = cv2.VideoCapture(str(example_video_path))
    frame_idx = 0
    max_frames = 10  # Extract only 10 frames for a quick test
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        
        # Save frame
        frame_name = f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(img_dir / frame_name), frame)
        
        # Create a dummy palette mask for the video frame (Object ID = 1)
        # Background is 0, Object is 1
        mask = np.zeros((h, w), dtype=np.uint8)
        # Moving rectangle to simulate tracking
        cx, cy = w // 2 + (frame_idx * 5), h // 2
        cv2.rectangle(mask, (cx - 30, cy - 30), (cx + 30, cy + 30), 1, -1)
        
        mask_name = f"{frame_idx:05d}.png"
        cv2.imwrite(str(ann_dir / mask_name), mask)
        
        frame_idx += 1
        
    cap.release()
    print(f"Stage 2 Mini Dataset created at {stage2_dir} ({frame_idx} frames)")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    example_img = project_root / "examples" / "sf.jpg"
    example_vid = project_root / "examples" / "videos" / "car.mp4"
    
    print("Creating a mini dataset using existing example files. No download required!")
    create_mini_stage1(current_dir, example_img)
    create_mini_stage2(current_dir, example_vid)
    
    print("\n=========================================================")
    print("Mini Dataset generation complete!")
    print(f"Stage 1 path: data/mini_SA1B")
    print(f"Stage 2 path: data/mini_DAVIS")
    print("\nYou can test training using these paths. Example:")
    print("python -m training.train_image --config training/configs/train_image_ti.yaml --data-root data/mini_SA1B --output-dir runs/mini_image --overfit-one-batch")
    print("=========================================================")
