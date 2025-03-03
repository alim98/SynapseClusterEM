#!/bin/bash

# Set default values
RAW_BASE_DIR="../data/raw"
SEG_BASE_DIR="../data/seg"
ADD_MASK_BASE_DIR="../data/add_mask"
EXCEL_DIR="../data/excel"
OUTPUT_DIR="../outputs/notebook_analysis"
CHECKPOINT_PATH="../hemibrain_production.checkpoint"
BBOX_NAMES="bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7"
SEGMENTATION_TYPES="9 10"
ALPHAS="1.0"
BATCH_SIZE=4
NUM_WORKERS=2

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
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --bbox_names)
      BBOX_NAMES="$2"
      shift 2
      ;;
    --segmentation_types)
      SEGMENTATION_TYPES="$2"
      shift 2
      ;;
    --alphas)
      ALPHAS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the analysis script
python scripts/run_notebook_analysis.py \
  --raw_base_dir "$RAW_BASE_DIR" \
  --seg_base_dir "$SEG_BASE_DIR" \
  --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
  --excel_dir "$EXCEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --bbox_names $BBOX_NAMES \
  --segmentation_types $SEGMENTATION_TYPES \
  --alphas $ALPHAS \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS

echo "Analysis complete. Results saved to $OUTPUT_DIR" 