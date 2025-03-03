#!/bin/bash

# Set default values
RAW_BASE_DIR="../data/raw"
SEG_BASE_DIR="../data/seg"
ADD_MASK_BASE_DIR="../data/add_mask"
EXCEL_DIR="../data/excel"
OUTPUT_DIR="outputs/gif_visualization"
BBOX_NAME="bbox1"
SAMPLE_INDEX=0
SEGMENTATION_TYPE=1
ALPHA=1.0
GRAY_VALUE=0.6
FPS=10

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
    --bbox_name)
      BBOX_NAME="$2"
      shift 2
      ;;
    --sample_index)
      SAMPLE_INDEX="$2"
      shift 2
      ;;
    --segmentation_type)
      SEGMENTATION_TYPE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --gray_value)
      GRAY_VALUE="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
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

# Run the visualization script
python scripts/visualize_sample_as_gif.py \
  --raw_base_dir "$RAW_BASE_DIR" \
  --seg_base_dir "$SEG_BASE_DIR" \
  --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
  --excel_dir "$EXCEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --bbox_name "$BBOX_NAME" \
  --sample_index "$SAMPLE_INDEX" \
  --segmentation_type "$SEGMENTATION_TYPE" \
  --alpha "$ALPHA" \
  --gray_value "$GRAY_VALUE" \
  --fps "$FPS"

echo "Visualization complete. Results saved to $OUTPUT_DIR" 