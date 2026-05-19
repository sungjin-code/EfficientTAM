#!/bin/bash
set -e

# Data directory path (one level below the script location)
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$DATA_DIR"

echo "Setting up datasets in $DATA_DIR"
echo "Environment: Linux (using wget)"

# ---------------------------------------------------------
# Stage 2: DAVIS 2017 Dataset Preparation
# ---------------------------------------------------------
DAVIS_DIR="$DATA_DIR/DAVIS"
if [ ! -d "$DAVIS_DIR/JPEGImages" ]; then
    echo -e "\n[1/2] Downloading DAVIS 2017 TrainVal 480p dataset for Stage 2..."
    wget -c https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -O "$DATA_DIR/DAVIS-2017.zip"
    
    echo "Extracting DAVIS 2017..."
    unzip -q "$DATA_DIR/DAVIS-2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/DAVIS-2017.zip"
    echo "✅ DAVIS dataset ready at $DAVIS_DIR"
else
    echo -e "\n[1/2] ✅ DAVIS dataset already exists at $DAVIS_DIR"
fi


# ---------------------------------------------------------
# Stage 1: SA-1B (Mock) Dataset Preparation
# ---------------------------------------------------------
SA1B_DIR="$DATA_DIR/SA1B"
if [ ! -d "$SA1B_DIR/images" ]; then
    echo -e "\n[2/2] Preparing SA-1B style directory structure for Stage 1..."
    mkdir -p "$SA1B_DIR/images"
    mkdir -p "$SA1B_DIR/masks"
    
    echo "Creating a dummy SA-1B sample for smoke-testing..."
    # Create dummy image and mask using python
    python3 -c "
import numpy as np
from PIL import Image
import os
img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
Image.fromarray(img).save('$SA1B_DIR/images/img001.jpg')
mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[256:768, 256:768] = 255
Image.fromarray(mask).save('$SA1B_DIR/masks/img001_1.png')
"
    echo "✅ Dummy SA-1B dataset ready at $SA1B_DIR"
    echo "⚠️  NOTE: SA-1B requires manual download from Meta AI after agreeing to their license."
    echo "    For actual training, download the data and use the 'prepare_sa1b.py'"
    echo "    script in this directory to format and apply the dataset."
else
    echo -e "\n[2/2] ✅ SA-1B directory already exists at $SA1B_DIR"
fi

echo -e "\n========================================================="
echo "🎉 Dataset preparation complete!"
echo "Stage 1 (Image) Data Root: $SA1B_DIR"
echo "Stage 2 (Video) Data Root: $DAVIS_DIR"
echo ""
echo "[Training Commands Example]"
echo "Stage 1: python -m training.train_image --config training/configs/train_image_s.yaml --data-root data/SA1B --output-dir runs/image_s"
echo "Stage 2: python -m training.train_video --config training/configs/train_video_s.yaml --data-root data/DAVIS --output-dir runs/video_s"
echo "========================================================="
