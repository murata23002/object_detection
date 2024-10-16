#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

SCRIPT_PATH=$(realpath "$0")

CHECKOUT_DIR="checkout/checkout_1_$DATE"
mkdir -p "$CHECKOUT_DIR"
cp "$SCRIPT_PATH" "$CHECKOUT_DIR"

python train.py --train ./dataset/head_face_body_limb \
    --test ./dataset/head_face_body_limb \
    --val ./dataset/head_face_body_limb \
    --batch 6 \
    --epochs 100 \
    --tfilteName head_face_body_limb_$DATE.tflite \
    --checkout "$CHECKOUT_DIR"
