import pandas as pd
import os
import re

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the path to the Excel file
excel_path = os.path.join(script_dir, 'man.xlsx')

# Read the Excel file
df = pd.read_excel(excel_path)

print("-" * 80)
print("SYNAPSE DATA ANALYSIS BY BBOX")
print("-" * 80)

# Get features from first column
features = df.iloc[:, 0].dropna().tolist()
# Get synapse columns (columns starting from H)
synapse_columns = df.columns[7:].tolist()

# Create a dictionary to store bbox information
bbox_synapses = {}

# First row contains the synapse names
synapse_names = df.iloc[0, 7:].values

# Extract the bbox number from each synapse name using regex
synapse_to_bbox = {}
for i, name in enumerate(synapse_names):
    if pd.isna(name):
        continue
        
    # Extract the numerical part from the synapse name - this will be the column index
    col_idx = 7 + i  # 7 is the offset for the first synapse column
    
    # The first row typically has column numbers which may indicate bbox
    # Let's get the numeric value from the column header
    bbox_num = synapse_columns[i]
    if isinstance(bbox_num, (int, float)) and not pd.isna(bbox_num):
        bbox = int(bbox_num)
    else:
        # If column header is not numeric, try to extract a number from it
        bbox_match = re.search(r'(\d+)', str(bbox_num))
        bbox = int(bbox_match.group(1)) if bbox_match else 0
    
    if bbox not in bbox_synapses:
        bbox_synapses[bbox] = []
    
    # Create data structure for this synapse
    synapse_data = {'synapse_name': name, 'features': {}}
    
    # Extract features for this synapse
    for feature in features:
        if feature != "synapse:":  # Skip the synapse name row
            feature_row_idx = df.index[df.iloc[:, 0] == feature].tolist()[0]
            value = df.iloc[feature_row_idx, col_idx]
            if pd.notna(value):
                synapse_data['features'][feature] = value
    
    bbox_synapses[bbox].append(synapse_data)

# Print organized data
for bbox_num in sorted(bbox_synapses.keys()):
    print(f"\nBbox {bbox_num}:")
    print("-" * 40)
    for synapse in bbox_synapses[bbox_num]:
        print(f"\nSynapse: {synapse['synapse_name']}")
        for feature, value in synapse['features'].items():
            print(f"  {feature}: {value}")

# Save organized data to CSV
csv_data = []
for bbox_num in sorted(bbox_synapses.keys()):
    for synapse in bbox_synapses[bbox_num]:
        row = {
            'bbox': bbox_num,
            'synapse_name': synapse['synapse_name']
        }
        row.update(synapse['features'])
        csv_data.append(row)

organized_df = pd.DataFrame(csv_data)
csv_path = os.path.join(script_dir, 'cleaned_df.csv')
organized_df.to_csv(csv_path, index=False)
print(f"\nReorganized data saved to {csv_path}")

# Print feature categories
print("\nFeature Categories (from columns B-G):")
print("-" * 50)
for i, feature in enumerate(features):
    if i < len(df) and feature != "synapse:":
        hint_values = df.iloc[i, 1:7].dropna().tolist()
        if hint_values:
            print(f"{feature}: {', '.join(str(x) for x in hint_values)}")

