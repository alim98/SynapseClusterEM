#!/bin/bash
# Model Input Visualization Tool Runner
# This shell script runs the model input visualization tool

echo "Model Input Visualization Tool"
echo "============================"
echo
echo "This tool visualizes the exact inputs that the neural network model receives."
echo

# Check if help parameter is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "------"
    echo "./run_visualization.sh [options]"
    echo
    echo "Examples:"
    echo "  ./run_visualization.sh                                  - Run with default settings"
    echo "  ./run_visualization.sh --num_samples 10                 - Visualize 10 random samples"
    echo "  ./run_visualization.sh --bbox_names bbox1 bbox2         - Only use bbox1 and bbox2"
    echo "  ./run_visualization.sh --segmentation_type 0            - Use raw data (no segmentation)"
    echo "  ./run_visualization.sh --fixed_gray_value 0.7           - Use 0.7 as the gray value"
    echo "  ./run_visualization.sh --alpha 0.5                      - Use alpha blending of 0.5"
    echo
    echo "All parameters:"
    echo "  --raw_base_dir PATH         - Base directory for raw data"
    echo "  --seg_base_dir PATH         - Base directory for segmentation data"
    echo "  --add_mask_base_dir PATH    - Base directory for additional mask data"
    echo "  --excel_dir PATH            - Directory containing Excel files"
    echo "  --output_dir PATH           - Directory to save output files"
    echo "  --bbox_names NAME1 NAME2... - Bounding box names to include"
    echo "  --num_samples NUM           - Number of samples to visualize"
    echo "  --sample_indices IDX1 IDX2  - Specific sample indices to visualize"
    echo "  --segmentation_type TYPE    - Segmentation type (0=raw, 1=presynapse, etc.)"
    echo "  --alpha VALUE               - Alpha value for blending (0.0-1.0)"
    echo "  --fixed_gray_value VALUE    - Fixed gray value for non-segmented regions (0.0-1.0)"
    echo "  --use_global_norm           - Use global normalization for visualization"
    exit 0
fi

# Default command
if [ $# -eq 0 ]; then
    echo "Running with default settings:"
    echo " - 100 samples"
    echo " - Segmentation type 1 (presynapse)"
    echo " - Alpha 1.0"
    echo " - Gray value 0.6"
    echo " - All bboxes (bbox1, bbox2, bbox3, bbox4, bbox5, bbox6, bbox7)"
    echo
    python "$(dirname "$0")/visualize_inputs.py"
    echo
    echo "Visualization complete!"
    echo "Results are saved in the outputs/model_input_visualization directory"
    exit 0
fi

# Check if bbox_names are specified
FOUND_BBOX=0
BBOX_LIST=""
CAPTURE=0

for arg in "$@"; do
    if [ $CAPTURE -eq 1 ]; then
        if [[ $arg == --* ]]; then
            CAPTURE=0
        else
            if [ -z "$BBOX_LIST" ]; then
                BBOX_LIST="$arg"
            else
                BBOX_LIST="$BBOX_LIST, $arg"
            fi
        fi
    fi
    
    if [ "$arg" == "--bbox_names" ]; then
        FOUND_BBOX=1
        CAPTURE=1
    fi
done

# Run with provided parameters
echo "Running with custom parameters..."
if [ $FOUND_BBOX -eq 1 ]; then
    echo " - Using bboxes: $BBOX_LIST"
else
    echo " - Using all bboxes (bbox1, bbox2, bbox3, bbox4, bbox5, bbox6, bbox7)"
fi
echo

python "$(dirname "$0")/visualize_inputs.py" "$@"

echo
echo "Visualization complete!"
echo "Results are saved in the outputs/model_input_visualization directory" 