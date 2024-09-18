#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S) 

python train.py --train ./path/to/dataset \
    --test ./path/to/dataset \
    --val ./path/to/dataset \
    --batch 4 \
    --epochs 500 \
    --tfilteName head_face_body_limb_$DATE.tflite \
    --checkout output/checkout_1_$DATE
