"""
Script to randomly sample 50 rows from each CSV file in the specified directories.
"""

import os
import pandas as pd
import glob

def process_csv_file(file_path, n_samples=100, seed=42):
    """
    Read a CSV file, randomly sample n_samples rows, and save it back.
    
    Args:
        file_path (str): Path to the CSV file
        n_samples (int): Number of rows to keep
        seed (int): Random seed for reproducibility
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if we have more rows than needed
        if len(df) > n_samples:
            # Randomly sample n_samples rows
            df_sampled = df.sample(n=n_samples, random_state=seed)
            
            # Save back to the same file
            df_sampled.to_csv(file_path, index=False)
            print(f"Successfully processed {file_path}")
            print(f"Reduced from {len(df)} to {len(df_sampled)} rows")
        else:
            print(f"Skipping {file_path} - has {len(df)} rows (less than or equal to {n_samples})")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Base directory
    base_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100"
    
    # Find all CSV files in all subfolders
    csv_pattern = os.path.join(base_dir, "**", "features_*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    for file_path in csv_files:
        print(f"\nProcessing: {file_path}")
        process_csv_file(file_path)

if __name__ == "__main__":
    main() 