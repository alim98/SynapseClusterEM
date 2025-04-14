import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import re

# File paths
manual_path = r"C:\Users\alim9\Documents\codes\synapse2\manual2\cleaned_df.csv"
vgg_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\10\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"

# Create output directory for plots
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'compare_plots')
os.makedirs(plots_dir, exist_ok=True)

def load_data():
    """
    1. Load both csv files
    """
    try:
        # Use local file if VGG file is not found at the specified path
        local_vgg_path = os.path.join(script_dir, 'features_layer20_seg10_alpha1_0.csv')
        
        # Load manual data
        if not os.path.exists(manual_path):
            print(f"Error: Manual file not found at {manual_path}")
            return None, None
        
        manual_df = pd.read_csv(manual_path)
        print(f"Loaded manual data: {manual_df.shape[0]} synapses with {manual_df.shape[1]} features")
        
        # Load VGG data
        vgg_df = None
        if os.path.exists(vgg_path):
            vgg_df = pd.read_csv(vgg_path)
            print(f"Loaded VGG data from original path: {vgg_df.shape[0]} samples with {vgg_df.shape[1]} features")
        elif os.path.exists(local_vgg_path):
            vgg_df = pd.read_csv(local_vgg_path)
            print(f"Loaded VGG data from local path: {vgg_df.shape[0]} samples with {vgg_df.shape[1]} features")
        else:
            print(f"Error: VGG file not found at {vgg_path} or {local_vgg_path}")
            return manual_df, None
        
        return manual_df, vgg_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None

def match_synapses(manual_df, vgg_df):
    """
    2. Find The synapses names from manual file in vgg file
    Match based on synapse names and convert bbox formats
    """
    try:
        if manual_df is None or vgg_df is None:
            print("Cannot match synapses: One or both dataframes are missing")
            return None, None
            
        # Hardcode column names based on the example provided
        manual_bbox_col = 'bbox'
        manual_name_col = 'synapse_name'
        vgg_bbox_col = 'bbox'
        vgg_name_col = 'Var1'
        
        print(f"Manual columns: {manual_df.columns.tolist()}")
        print(f"VGG columns: {vgg_df.columns.tolist()}")
        
        print(f"Using manual bbox column: {manual_bbox_col}, name column: {manual_name_col}")
        print(f"Using VGG bbox column: {vgg_bbox_col}, name column: {vgg_name_col}")
        
        # Create copies to avoid modifying the originals
        manual_df = manual_df.copy()
        vgg_df = vgg_df.copy()
        
        # Print a few rows from each dataframe for debugging
        print("\nFirst few rows of manual dataframe:")
        if manual_name_col in manual_df.columns and manual_bbox_col in manual_df.columns:
            print(manual_df[[manual_bbox_col, manual_name_col]].head())
        else:
            print("Warning: Could not find expected columns in manual dataframe")
            print(manual_df.head())
        
        print("\nFirst few rows of VGG dataframe:")
        if vgg_name_col in vgg_df.columns and vgg_bbox_col in vgg_df.columns:
            print(vgg_df[[vgg_bbox_col, vgg_name_col]].head())
        else:
            print("Warning: Could not find expected columns in VGG dataframe")
            print(vgg_df.head())
        
        # Extract the numeric part from VGG bbox (e.g., "bbox1" -> "1")
        vgg_df['bbox_num'] = vgg_df[vgg_bbox_col].astype(str).str.extract(r'bbox(\d+)', expand=False)
        
        # Convert manual bbox to string for comparison
        manual_df['bbox_str'] = manual_df[manual_bbox_col].astype(str)
        
        # Match based on synapse names and bbox numbers
        matched_pairs = []
        
        # Create dictionaries to store matches
        manual_to_vgg = {}  # Maps manual index to vgg index
        vgg_to_manual = {}  # Maps vgg index to manual index
        
        # Check exact matches first (both name and bbox)
        for manual_idx, manual_row in manual_df.iterrows():
            manual_synapse = str(manual_row[manual_name_col]).strip() if manual_name_col in manual_df.columns else ""
            manual_bbox = str(manual_row[manual_bbox_col]) if manual_bbox_col in manual_df.columns else ""
            
            for vgg_idx, vgg_row in vgg_df.iterrows():
                vgg_synapse = str(vgg_row[vgg_name_col]).strip() if vgg_name_col in vgg_df.columns else ""
                vgg_bbox_num = str(vgg_row['bbox_num']) if 'bbox_num' in vgg_df.columns else ""
                
                # Check for exact synapse name match AND bbox number match
                if manual_synapse == vgg_synapse and manual_bbox == vgg_bbox_num:
                    manual_to_vgg[manual_idx] = vgg_idx
                    vgg_to_manual[vgg_idx] = manual_idx
                    matched_pairs.append((manual_bbox, manual_synapse, vgg_bbox_num, vgg_synapse))
                    break
        
        # Create matched dataframes with reset indices to avoid indexing issues
        if manual_to_vgg:
            # Get matching rows but reset index to avoid issues with loc/iloc later
            matched_manual_df = manual_df.loc[list(manual_to_vgg.keys())].copy().reset_index(drop=True)
            matched_vgg_df = vgg_df.loc[list(vgg_to_manual.keys())].copy().reset_index(drop=True)
            
            # Add matching index columns for later reference
            matched_manual_df['original_index'] = list(manual_to_vgg.keys())
            matched_vgg_df['original_index'] = list(vgg_to_manual.keys())
            
            # Ensure rows are in the same order in both dataframes
            # This is critical to make sure we can use integer indices to match between them
            matched_manual_df = matched_manual_df.sort_values('original_index').reset_index(drop=True)
            matched_vgg_df = matched_vgg_df.sort_values('original_index').reset_index(drop=True)
        else:
            matched_manual_df = None
            matched_vgg_df = None
        
        # Print matching results
        print(f"\nFound {len(matched_pairs)} matching synapses")
        if len(matched_pairs) > 0:
            print("\nMatched pairs (manual_bbox, manual_synapse, vgg_bbox_num, vgg_synapse):")
            for pair in matched_pairs[:10]:  # Show first 10 matches
                print(f"  {pair}")
        else:
            print("\nNo matches found. Let's try with more flexible matching...")
            
            # Try more flexible matching if no exact matches found
            manual_to_vgg = {}
            vgg_to_manual = {}
            matched_pairs = []
            
            for manual_idx, manual_row in manual_df.iterrows():
                manual_synapse = str(manual_row[manual_name_col]).strip() if manual_name_col in manual_df.columns else ""
                
                for vgg_idx, vgg_row in vgg_df.iterrows():
                    vgg_synapse = str(vgg_row[vgg_name_col]).strip() if vgg_name_col in vgg_df.columns else ""
                    
                    # Match only on synapse name
                    if manual_synapse == vgg_synapse and len(manual_synapse) > 0:
                        manual_to_vgg[manual_idx] = vgg_idx
                        vgg_to_manual[vgg_idx] = manual_idx
                        matched_pairs.append((manual_synapse, vgg_synapse))
                        break
            
            # Create matched dataframes with reset indices
            if manual_to_vgg:
                matched_manual_df = manual_df.loc[list(manual_to_vgg.keys())].copy().reset_index(drop=True)
                matched_vgg_df = vgg_df.loc[list(vgg_to_manual.keys())].copy().reset_index(drop=True)
                
                # Add matching index columns for later reference
                matched_manual_df['original_index'] = list(manual_to_vgg.keys())
                matched_vgg_df['original_index'] = list(vgg_to_manual.keys())
                
                # Ensure rows are in the same order
                matched_manual_df = matched_manual_df.sort_values('original_index').reset_index(drop=True)
                matched_vgg_df = matched_vgg_df.sort_values('original_index').reset_index(drop=True)
            else:
                matched_manual_df = None
                matched_vgg_df = None
            
            print(f"Found {len(matched_pairs)} matching synapses with flexible matching")
            if len(matched_pairs) > 0:
                print("\nMatched pairs (manual_synapse, vgg_synapse):")
                for pair in matched_pairs[:10]:  # Show first 10 matches
                    print(f"  {pair}")
        
        return matched_manual_df, matched_vgg_df
    
    except Exception as e:
        print(f"Error matching synapses: {e}")
        traceback.print_exc()
        return None, None

def compute_umap(vgg_df):
    """
    3. Compute the vgg file umap in 2d
    """
    try:
        if vgg_df is None:
            print("Cannot compute UMAP: VGG dataframe is missing")
            return None
        
        # Check if UMAP coordinates are already present
        if 'umap_x' in vgg_df.columns and 'umap_y' in vgg_df.columns:
            print("UMAP coordinates already present in VGG data, using existing coordinates")
            return vgg_df
        
        # Extract numerical features from VGG dataframe
        # Get columns that start with 'layer' which typically contain numerical features
        feature_cols = [col for col in vgg_df.columns if col.startswith('layer')]
        
        # If no layer columns found, try to guess other numerical columns
        if not feature_cols:
            feature_cols = [col for col in vgg_df.columns 
                           if col not in ['bbox', 'Var1', 'bbox_name', 'bbox_normalized', 'synapse_normalized', 
                                         'bbox_numeric', 'central_coord_1', 'central_coord_2', 'central_coord_3',
                                         'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3',
                                         'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3']]
        
        print(f"Using {len(feature_cols)} feature columns for UMAP")
        
        # Create matrix of feature values
        X = vgg_df[feature_cols].values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = reducer.fit_transform(X_scaled)
        
        # Add UMAP coordinates to the dataframe
        vgg_df = vgg_df.copy()
        vgg_df['umap_x'] = umap_result[:, 0]
        vgg_df['umap_y'] = umap_result[:, 1]
        
        print(f"UMAP computed successfully for VGG data")
        return vgg_df
    
    except Exception as e:
        print(f"Error computing UMAP: {e}")
        traceback.print_exc()
        return None

def create_color_visualization(matched_manual_df, vgg_umap_df, feature):
    """
    4-8. Plot the umap 2d of vgg file and color the points by the manual file features
    with distinct colors for each category
    """
    try:
        if matched_manual_df is None or vgg_umap_df is None:
            print(f"Cannot create visualization for {feature}: Missing data")
            return
        
        if feature not in matched_manual_df.columns:
            print(f"Feature '{feature}' not found in manual data")
            return
        
        plt.figure(figsize=(14, 12))
        
        # Get unique values directly from the data
        unique_values = matched_manual_df[feature].dropna().unique()
        print(f"Creating visualization for feature '{feature}' with {len(unique_values)} unique values: {unique_values}")
        
        # Define ordinal features and their expected order
        ordinal_features = {
            'vesicle size': ['small', 'medium', 'large'],
            'shape (roundness)': None,  # Not ordinal
            'Shading inside large vesicles': None,  # Not ordinal
            'Presynaptic density (PSD) size - shading around the presynaptic ': ['small', 'medium', 'large'],
            'Location (on spines, dendrites)': None,  # Not ordinal
            'size of presyn compartment': ['thin', 'medium', 'thick'],
            'size of postsyn compartment': ['thin', 'medium', 'thick'],
            'single synapse or dyad (or >)': ['single', 'dyad', 'multi'],
            'cleft thickness (how pronounced/obvious)': ['thin', 'medium', 'thick'],
            'size of the vesicle cloud': ['small', 'medium', 'large'],
            'packing density': ['low', 'medium', 'high'],
            'number of docked vesicles': ['few', 'some', 'many'],
            'mitochondria close by (<300nm from cleft)?': None  # Binary, not ordinal
        }
        
        # Define color gradients for ordinal features
        color_gradients = {
            'size': ['#66c2a5', '#8da0cb', '#fc8d62'],  # small -> medium -> large
            'thickness': ['#66c2a5', '#8da0cb', '#fc8d62'],  # thin -> medium -> thick
            'density': ['#66c2a5', '#8da0cb', '#fc8d62'],  # low -> medium -> high
            'count': ['#66c2a5', '#8da0cb', '#fc8d62']  # few -> some -> many
        }
        
        # Define distinct colors for non-ordinal features
        distinct_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', 
            '#f58231', '#911eb4', '#42d4f4', '#f032e6',
            '#bfef45', '#fabed4', '#469990', '#dcbeff',
            '#9A6324', '#fffac8', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000075', '#a9a9a9'
        ]
        
        # Determine whether this feature is ordinal and get color mapping
        is_ordinal = feature in ordinal_features and ordinal_features[feature] is not None
        
        # Create the color map
        colors = {}
        
        if is_ordinal:
            # Use gradient colors for ordinal features
            expected_order = ordinal_features[feature]
            
            # Determine which gradient to use
            if 'size' in feature.lower() or 'large' in str(unique_values).lower() or 'small' in str(unique_values).lower():
                gradient = color_gradients['size']
            elif 'thick' in str(unique_values).lower() or 'thin' in str(unique_values).lower():
                gradient = color_gradients['thickness']
            elif 'density' in feature.lower() or 'low' in str(unique_values).lower() or 'high' in str(unique_values).lower():
                gradient = color_gradients['density']
            elif 'few' in str(unique_values).lower() or 'many' in str(unique_values).lower() or 'docked' in feature.lower():
                gradient = color_gradients['count']
            else:
                gradient = color_gradients['size']  # default
            
            # Make sure we have enough colors in the gradient
            if len(gradient) < len(expected_order):
                # Extend the gradient by interpolating
                extended_gradient = []
                for i in range(len(expected_order)):
                    idx = i * (len(gradient) - 1) / (len(expected_order) - 1) if len(expected_order) > 1 else 0
                    lower_idx = int(idx)
                    upper_idx = min(lower_idx + 1, len(gradient) - 1)
                    frac = idx - lower_idx
                    
                    # Convert hex to rgb, interpolate, then back to hex
                    lower_rgb = np.array(mcolors.hex2color(gradient[lower_idx]))
                    upper_rgb = np.array(mcolors.hex2color(gradient[upper_idx]))
                    interp_rgb = lower_rgb * (1 - frac) + upper_rgb * frac
                    extended_gradient.append(mcolors.rgb2hex(interp_rgb))
                
                gradient = extended_gradient
            
            # Map expected values to their gradient colors
            for i, val in enumerate(expected_order):
                # Find the value in our unique values that best matches
                for unique_val in unique_values:
                    if str(unique_val).lower() == val.lower() or str(unique_val).lower().startswith(val.lower()):
                        colors[unique_val] = gradient[i]
            
            # Handle any missing values
            for val in unique_values:
                if val not in colors:
                    # If it contains any of our expected values as a substring, use that color
                    matched = False
                    for i, expected_val in enumerate(expected_order):
                        if expected_val.lower() in str(val).lower():
                            colors[val] = gradient[i]
                            matched = True
                            break
                    
                    # If no match, use a gray color
                    if not matched:
                        colors[val] = '#a0a0a0'  # gray
                        
        else:
            # For non-ordinal features, use distinct colors
            if len(unique_values) > len(distinct_colors):
                # If more values than colors, cycle through colors
                extended_colors = []
                while len(extended_colors) < len(unique_values):
                    extended_colors.extend(distinct_colors)
                distinct_colors = extended_colors[:len(unique_values)]
            
            # Create the color map with distinct colors
            colors = {val: distinct_colors[i] for i, val in enumerate(unique_values)}
        
        # Get ALL VGG points - this should include both matched and unmatched
        all_vgg_points = None
        
        # If vgg_umap_df is the subset with only matched points, try to get the original full dataset
        if 'original_index' in vgg_umap_df.columns:
            # The current vgg_umap_df might be only the matched points
            # We need to reload the full VGG data with UMAP coordinates
            try:
                # Use local file if VGG file is not found at the specified path
                local_vgg_path = os.path.join(script_dir, 'features_layer20_seg10_alpha1_0.csv')
                
                if os.path.exists(vgg_path):
                    all_vgg_points = pd.read_csv(vgg_path)
                elif os.path.exists(local_vgg_path):
                    all_vgg_points = pd.read_csv(local_vgg_path)
                
                if all_vgg_points is not None and ('umap_x' in all_vgg_points.columns and 'umap_y' in all_vgg_points.columns):
                    print(f"Using full VGG dataset with {len(all_vgg_points)} points")
                else:
                    all_vgg_points = None
            except Exception as e:
                print(f"Could not load full VGG dataset: {e}")
                all_vgg_points = None
        
        # If we couldn't get the full dataset, use what we have
        if all_vgg_points is None:
            all_vgg_points = vgg_umap_df
            print("Using only matched VGG points")
        
        # First, plot ALL VGG points in gray (9. all other synapses should be in gray)
        plt.scatter(all_vgg_points['umap_x'], all_vgg_points['umap_y'], 
                   color='lightgray', alpha=0.5, s=30, edgecolors='none',
                   label='Unmatched synapses')
        
        # Create a mask to track which points have been colored
        colored_points = set()
        
        # Plot matched points with their feature-based colors
        for value in unique_values:
            # Find manual samples with this feature value
            value_mask = matched_manual_df[feature] == value
            if not any(value_mask):
                continue
            
            # Find corresponding VGG points using the mask
            vgg_subset = vgg_umap_df[value_mask].copy()
            
            if len(vgg_subset) == 0:
                continue
            
            # Keep track of which points we've colored
            for idx in vgg_subset.index:
                colored_points.add(idx)
                
            # Plot these points with their feature color
            plt.scatter(vgg_subset['umap_x'], vgg_subset['umap_y'],
                       color=colors[value], alpha=0.9, s=100, 
                       edgecolors='black', label=str(value))
        
        # Create legend with distinct colors
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[val], markersize=10,
                                 label=str(val)) for val in unique_values if any(matched_manual_df[feature] == val)]
        
        # Add unmatched points to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='lightgray', markersize=10,
                                     label='Unmatched synapses'))
        
        # Create a simpler title for display
        feature_title = feature.replace('?', '')
        
        plt.legend(handles=legend_elements, title=feature_title, 
                  loc='best', bbox_to_anchor=(1, 1), fontsize=12)
        
        plt.title(f'UMAP Visualization Colored by {feature_title}', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=14)
        plt.ylabel('UMAP Dimension 2', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the plot with properly sanitized filename
        # Replace invalid Windows filename characters: \ / : * ? " < > |
        safe_feature_name = feature.replace(" ", "_")
        safe_feature_name = safe_feature_name.replace("\\", "")
        safe_feature_name = safe_feature_name.replace("/", "_")
        safe_feature_name = safe_feature_name.replace(":", "")
        safe_feature_name = safe_feature_name.replace("*", "")
        safe_feature_name = safe_feature_name.replace("?", "")
        safe_feature_name = safe_feature_name.replace("\"", "")
        safe_feature_name = safe_feature_name.replace("<", "lt")
        safe_feature_name = safe_feature_name.replace(">", "gt")
        safe_feature_name = safe_feature_name.replace("|", "_")
        safe_feature_name = safe_feature_name.replace("(", "")
        safe_feature_name = safe_feature_name.replace(")", "")
        
        filename = f'umap_colored_by_{safe_feature_name}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()
        
        print(f"Created visualization for feature: {feature}")
    
    except Exception as e:
        print(f"Error creating visualization for {feature}: {e}")
        traceback.print_exc()

def map_feature_values(manual_df):
    """
    Map feature values to standardized categories based on the TODO list
    """
    try:
        if manual_df is None:
            print("Cannot map feature values: Manual dataframe is missing")
            return None
        
        # Create a copy to avoid modifying the original
        df = manual_df.copy()
        
        # List of features from the TODO section
        todo_features = [
            'vesicle size', 
            'shape (roundness)', 
            'Shading inside large vesicles',
            'Presynaptic density (PSD) size - shading around the presynaptic ',
            'Location (on spines, dendrites)', 
            'size of presyn compartment',
            'size of postsyn compartment', 
            'single synapse or dyad (or >)',
            'cleft thickness (how pronounced/obvious)', 
            'size of the vesicle cloud',
            'packing density', 
            'number of docked vesicles', 
            'mitochondria close by (<300nm from cleft)?'
        ]
        
        # Process each feature in the dataframe that matches our TODO list
        for feature in todo_features:
            if feature not in df.columns:
                print(f"Feature '{feature}' not found in dataframe, skipping")
                continue
            
            # Get unique values for this feature and print them
            unique_values = df[feature].dropna().unique()
            print(f"Feature '{feature}' has {len(unique_values)} unique values: {unique_values}")
            
            # We don't need to modify the values, just display what's in the data
            
        print("Feature value display complete")
        return df
    
    except Exception as e:
        print(f"Error displaying feature values: {e}")
        traceback.print_exc()
        return manual_df

def main():
    print("Synapse Comparison Tool")
    print("-" * 40)
    
# 1. Load both csv files
    print("Loading data files...")
    manual_df, vgg_df = load_data()
    
    if manual_df is None:
        print("Analysis aborted: Manual data file could not be loaded")
        return
        
    if vgg_df is None:
        print("Analysis aborted: VGG data file could not be loaded")
        return
    
    # 2. Find the synapse names from manual file in vgg file
    print("\nMatching synapses between files...")
    matched_manual_df, matched_vgg_df = match_synapses(manual_df, vgg_df)
    
    if matched_manual_df is None or matched_vgg_df is None or len(matched_manual_df) == 0:
        print("Analysis aborted: No matching synapses found")
        return
    
    # Map feature values to standardized categories
    print("\nMapping feature values to standardized categories...")
    matched_manual_df = map_feature_values(matched_manual_df)
    
    # 3. Compute the vgg file umap in 2d
    print("\nComputing UMAP for VGG features...")
    vgg_umap_df = compute_umap(matched_vgg_df)
    
    if vgg_umap_df is None:
        print("Analysis aborted: UMAP computation failed")
        return
    
    # 4-8. Plot the umap for each feature in the manual file
    print("\nCreating visualizations for each feature...")
    
    # List of features to visualize
    features_to_plot = [
        'vesicle size', 
        'shape (roundness)', 
        'Shading inside large vesicles',
        'Presynaptic density (PSD) size - shading around the presynaptic ',
        'Location (on spines, dendrites)', 
        'size of presyn compartment',
        'size of postsyn compartment', 
        'single synapse or dyad (or >)',
        'cleft thickness (how pronounced/obvious)', 
        'size of the vesicle cloud',
        'packing density', 
        'number of docked vesicles', 
        'mitochondria close by (<300nm from cleft)?'
    ]
    
    # Filter to only include features that are in the dataframe
    available_features = [f for f in features_to_plot if f in matched_manual_df.columns]
    
    for feature in available_features:
        print(f"  Processing feature: {feature}")
        create_color_visualization(matched_manual_df, vgg_umap_df, feature)
    
    print("\nAnalysis complete. Visualizations saved to:", plots_dir)

if __name__ == "__main__":
    main()
