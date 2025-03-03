#!/bin/bash

# SynapseClusterEM Complete Workflow Script
# This script runs the complete analysis pipeline for 3D synapse structures

# Default values - MODIFY THESE FOR YOUR ENVIRONMENT
RAW_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/raw"
SEG_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/seg"
ADD_MASK_BASE_DIR="/Users/ali/Documents/MPG_Doc/code/vesicle_cloud__syn_interface__mitochondria_annotation"
EXCEL_DIR="/Users/ali/Documents/MPG_Doc/code/7_bboxes_plus_seg/"
CHECKPOINT_PATH="/path/to/vgg3d_checkpoint.pth"  # MODIFY THIS
OUTPUT_DIR="outputs/workflow_results"
BBOX_NAMES="bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7"
SEGMENTATION_TYPES="1 2 3"
ALPHAS="1.0"
BATCH_SIZE=4
NUM_WORKERS=2
N_CLUSTERS=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "SynapseClusterEM Analysis Pipeline"
echo "========================================================"

# Step 1: Calculate global normalization statistics
echo ""
echo "Step 1: Calculating global normalization statistics..."
echo "------------------------------------------------------"

python scripts/global_norm_example.py \
    --raw_base_dir "$RAW_BASE_DIR" \
    --seg_base_dir "$SEG_BASE_DIR" \
    --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
    --excel_dir "$EXCEL_DIR" \
    --output_dir "$OUTPUT_DIR/global_norm" \
    --segmentation_type 1

GLOBAL_STATS_PATH="$OUTPUT_DIR/global_norm/global_stats.json"

# Step 2: Run the main analysis with global normalization
echo ""
echo "Step 2: Running feature extraction, clustering, and visualization..."
echo "------------------------------------------------------"

python scripts/run_analysis.py \
    --raw_base_dir "$RAW_BASE_DIR" \
    --seg_base_dir "$SEG_BASE_DIR" \
    --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
    --excel_dir "$EXCEL_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --bbox_names $BBOX_NAMES \
    --segmentation_types $SEGMENTATION_TYPES \
    --alphas $ALPHAS \
    --output_dir "$OUTPUT_DIR/analysis" \
    --use_global_norm \
    --global_stats_path "$GLOBAL_STATS_PATH" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --n_clusters $N_CLUSTERS

# Step 3: Visualize samples from each cluster
echo ""
echo "Step 3: Visualizing representative samples from each cluster..."
echo "------------------------------------------------------"

# Find the first segmentation type result directory
SEG_DIR=$(find "$OUTPUT_DIR/analysis" -type d -name "seg*" | head -n 1)

if [ -n "$SEG_DIR" ]; then
    # Read the cluster assignments
    CLUSTER_CSV="$SEG_DIR/features_with_clusters.csv"
    
    if [ -f "$CLUSTER_CSV" ]; then
        # For each cluster, visualize a representative sample
        for CLUSTER in $(seq 0 $((N_CLUSTERS-1))); do
            # Get a sample from this cluster (using awk to find first occurrence)
            SAMPLE_INFO=$(awk -F, -v cluster="$CLUSTER" '$NF == cluster {print $0; exit}' "$CLUSTER_CSV")
            
            if [ -n "$SAMPLE_INFO" ]; then
                # Extract bbox_name and index
                BBOX_NAME=$(echo "$SAMPLE_INFO" | awk -F, '{print $2}')
                SAMPLE_INDEX=$(echo "$SAMPLE_INFO" | awk -F, '{print $1}')
                
                echo "Visualizing sample from cluster $CLUSTER (bbox: $BBOX_NAME, index: $SAMPLE_INDEX)"
                
                # Run visualization for this sample
                for SEG_TYPE in $SEGMENTATION_TYPES; do
                    python scripts/visualize_sample_as_gif.py \
                        --raw_base_dir "$RAW_BASE_DIR" \
                        --seg_base_dir "$SEG_BASE_DIR" \
                        --add_mask_base_dir "$ADD_MASK_BASE_DIR" \
                        --excel_dir "$EXCEL_DIR" \
                        --bbox_name "$BBOX_NAME" \
                        --sample_index "$SAMPLE_INDEX" \
                        --segmentation_type "$SEG_TYPE" \
                        --alpha 1.0 \
                        --gray_value 0.5 \
                        --fps 10 \
                        --output_dir "$OUTPUT_DIR/visualizations/cluster_${CLUSTER}"
                done
            fi
        done
    else
        echo "Cluster CSV file not found: $CLUSTER_CSV"
    fi
else
    echo "No segmentation results found in $OUTPUT_DIR/analysis"
fi

echo ""
echo "Analysis pipeline completed!"
echo "Results are available in: $OUTPUT_DIR"
echo "========================================================" 