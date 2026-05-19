import os
import json
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("Error: pycocotools is required. Please install it using 'pip install pycocotools'")
    exit(1)

def convert_sa1b(input_dir, output_dir):
    """
    Converts SA-1B dataset (.jpg and .json) into EfficientTAM Stage 1 format.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .json files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    print(f"Found {len(json_files)} JSON files. Starting conversion...")
    
    for json_path in json_files:
        # Find corresponding image file (usually same name as .json but .jpg)
        img_name = json_path.stem + ".jpg"
        img_path = input_dir / img_name
        
        if not img_path.exists():
            print(f"Warning: Image file not found for {json_path.name}")
            continue
            
        # 1. Copy image
        shutil.copy2(img_path, img_out_dir / img_name)
        
        # 2. Convert and save masks
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        annotations = data.get('annotations', [])
        
        for idx, ann in enumerate(annotations):
            # Generate mask via RLE decoding
            rle = ann['segmentation']
            binary_mask = maskUtils.decode(rle)
            
            # Scale mask values (0 or 255)
            # Assuming SA-1B format requires binary (0 or 255) pixel values
            mask_img = (binary_mask * 255).astype(np.uint8)
            
            mask_filename = f"{json_path.stem}_{idx + 1}.png"
            cv2.imwrite(str(mask_out_dir / mask_filename), mask_img)
            
    print(f"Successfully converted SA-1B data to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SA-1B dataset for EfficientTAM Stage 1")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to extracted SA-1B tar files containing .jpg and .json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output data_root for EfficientTAM")
    
    args = parser.parse_args()
    convert_sa1b(args.input_dir, args.output_dir)
