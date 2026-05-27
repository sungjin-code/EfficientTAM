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
    print(
        "Error: pycocotools is required. Please install it using 'pip install pycocotools'"
    )
    exit(1)


def convert_sa1b(input_dir, output_dir, max_images=None, progress_every=100):
    """
    Converts SA-1B dataset (.jpg and .json) into EfficientTAM Stage 1 format.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"

    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all .json files. SA-1B shards may be extracted either flat or with
    # one subdirectory per tar shard.
    json_files = list(input_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    if max_images is not None:
        json_files = json_files[:max_images]

    print(f"Found {len(json_files)} JSON files. Starting conversion...")

    converted = 0
    skipped = 0
    for idx_json, json_path in enumerate(json_files, start=1):
        # Find corresponding image file (usually same name as .json but .jpg)
        img_name = json_path.stem + ".jpg"
        img_path = json_path.with_name(img_name)

        if not img_path.exists():
            print(f"Warning: Image file not found for {json_path.name}")
            skipped += 1
            continue

        # 1. Copy image
        shutil.copy2(img_path, img_out_dir / img_name)

        # 2. Convert and save masks
        with open(json_path, "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])

        for idx, ann in enumerate(annotations):
            # Generate mask via RLE decoding
            rle = ann["segmentation"]
            binary_mask = maskUtils.decode(rle)

            # Scale mask values (0 or 255)
            # Assuming SA-1B format requires binary (0 or 255) pixel values
            mask_img = (binary_mask * 255).astype(np.uint8)

            mask_filename = f"{json_path.stem}_{idx + 1}.png"
            cv2.imwrite(str(mask_out_dir / mask_filename), mask_img)

        converted += 1
        if progress_every > 0 and idx_json % progress_every == 0:
            print(f"Converted {idx_json}/{len(json_files)} JSON files...")

    print(
        f"Successfully converted SA-1B data to {output_dir} "
        f"(converted={converted}, skipped={skipped})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SA-1B dataset for EfficientTAM Stage 1"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to extracted SA-1B tar files containing .jpg and .json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output data_root for EfficientTAM",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Convert at most this many image/json pairs.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N JSON files. Use 0 to disable.",
    )

    args = parser.parse_args()
    convert_sa1b(
        args.input_dir,
        args.output_dir,
        max_images=args.max_images,
        progress_every=args.progress_every,
    )
