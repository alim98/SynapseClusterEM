#!/bin/bash

# Default values
RAW_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/raw"
SEG_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/seg"
ADD_MASK_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/vesicle_cloud__syn_interface__mitochondria_annotation"
EXCEL_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/"
OUTPUT_DIR="outputs/global_norm"
SEGMENTATION_TYPE=1  # Use segmentation type 1 (presynapse)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw_base_dir)
            RAW_BASE_DIR="$2"
            shift 2
            ;;
        --seg_base_dir)
            SEG_BASE_DIR="$2"
            shift 2
            ;;
        --add_mask_base_dir)
            ADD_MASK_BASE_DIR="$2"
            shift 2
            ;;
        --excel_dir)
            EXCEL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --segmentation_type)
            SEGMENTATION_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Make the script executable
chmod +x scripts/global_norm_example.py

# Run the global normalization example
python scripts/global_norm_example.py \
    --raw_base_dir "$RAW_BASE_DIR" \
    --seg_base_dir "$SEG_BASE_DIR" \
    --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
    --excel_dir "$EXCEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --segmentation_type "$SEGMENTATION_TYPE" 