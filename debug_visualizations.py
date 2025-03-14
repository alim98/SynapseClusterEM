"""
Debug script to test visualization consolidation and indexing.

This script finds the most recent run folder and attempts to create a visualization index.
"""

import os
import glob
import sys
from synapse_pipeline import SynapsePipeline
from synapse import config

def find_latest_run_folder():
    """Find the most recent run folder in the results directory."""
    # Look in the results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        return None
    
    # Find all run_* folders
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))
    if not run_folders:
        print(f"Error: No run folders found in '{results_dir}'.")
        return None
    
    # Sort by modification time (most recent first)
    run_folders.sort(key=os.path.getmtime, reverse=True)
    
    latest_folder = run_folders[0]
    print(f"Found latest run folder: {latest_folder}")
    return latest_folder

def count_image_files(directory):
    """Count image files in a directory recursively."""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"]
    count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    
    return count

def main():
    """Main function to debug visualization indexing."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Debug visualization indexing")
    parser.add_argument('run_folder', nargs='?', help='Run folder to analyze (e.g., run_2025-03-14_12-00-51)')
    parser.add_argument('--force-consolidate', action='store_true', help='Force reconsolidation of visualizations')
    parser.add_argument('--disable-gifs', action='store_true', help='Disable GIF generation during any analysis')
    parser.add_argument('--structured', action='store_true', help='Create structured visualizations')
    
    args, remaining_args = parser.parse_known_args()
    
    # Let the config parse the rest of the arguments
    import sys
    sys.argv = [sys.argv[0]] + remaining_args
    config.parse_args()
    
    # Find the latest run folder
    latest_folder = find_latest_run_folder()
    if not latest_folder:
        print("Exiting: No run folder found.")
        return
    
    # Check if we were provided with a specific run folder
    if args.run_folder and args.run_folder.startswith("run_"):
        manual_folder = os.path.join("results", args.run_folder)
        if os.path.exists(manual_folder) and os.path.isdir(manual_folder):
            latest_folder = manual_folder
            print(f"Using manually specified run folder: {latest_folder}")
    
    # Create pipeline instance
    pipeline = SynapsePipeline(config)
    
    # Manually set the results parent directory
    pipeline.results_parent_dir = latest_folder
    
    # List all files and directories in the run folder
    print("\nFiles and directories in the run folder:")
    for root, dirs, files in os.walk(latest_folder):
        level = root.replace(latest_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        
        # Only show files in the immediate directory for brevity
        if level < 2:  # Only show top level and one level down
            for f in files:
                print(f"{sub_indent}{f}")
        else:
            file_count = len(files)
            if file_count > 0:
                print(f"{sub_indent}... ({file_count} files)")
    
    # Check for special analysis results
    vesicle_analysis_dir = os.path.join(latest_folder, "vesicle_size_analysis")
    vesicle_html_report = os.path.join(vesicle_analysis_dir, "vesicle_size_analysis.html")
    
    if os.path.exists(vesicle_html_report):
        print(f"\nFound vesicle size analysis report: {vesicle_html_report}")
        
        # Count HTML reports in vesicle analysis directory
        html_files = glob.glob(os.path.join(vesicle_analysis_dir, "*.html"))
        print(f"Vesicle analysis directory contains {len(html_files)} HTML reports:")
        for html_file in html_files:
            print(f"  - {os.path.basename(html_file)}")
    
    # Check if structured visualizations already exist
    structured_dir = os.path.join(latest_folder, "structured_visualizations")
    structured_html = os.path.join(structured_dir, "visualization_structure.html")
    structured_exists = os.path.exists(structured_html)
    
    if structured_exists:
        print(f"\nFound existing structured visualization index: {structured_html}")
    
    # Create structured visualizations if requested or if they don't exist
    if args.structured or (not structured_exists and args.run_folder):
        print("\nCreating structured visualizations...")
        structured_vis_paths = pipeline.create_structured_visualizations()
        if structured_vis_paths and "index_html" in structured_vis_paths:
            structured_html = structured_vis_paths["index_html"]
            print(f"Created structured visualization index at: {structured_html}")
    
    # Check if consolidated directory already exists and has files
    consolidated_dir = os.path.join(latest_folder, "all_visualizations")
    visualizations_already_consolidated = False
    
    if os.path.exists(consolidated_dir):
        consolidated_file_count = count_image_files(consolidated_dir)
        print(f"\nFound existing consolidated directory with {consolidated_file_count} visualization files.")
        if consolidated_file_count > 0:
            visualizations_already_consolidated = True
    
    # Force reconsolidation if requested
    if args.force_consolidate:
        visualizations_already_consolidated = False
        print("Forcing reconsolidation of visualizations...")
    
    # Try to consolidate visualizations if needed
    if not visualizations_already_consolidated:
        print("\nAttempting to consolidate visualizations...")
        consolidated_dir, copied_files = pipeline.consolidate_visualizations()
        
        if copied_files:
            print(f"\nCreated consolidated directory with {len(copied_files)} files.")
        else:
            print("\nNo new visualization files found or copied.")
    
    # Always create visualization index
    print("Creating visualization index...")
    html_path = pipeline.create_visualization_index()
    print(f"Visualization index created at: {html_path}")
    
    # Check if there are special analyses to show
    show_special_reports = []
    
    if os.path.exists(vesicle_html_report):
        show_special_reports.append(("Vesicle Size Analysis", vesicle_html_report))
    
    if structured_exists or args.structured:
        show_special_reports.append(("Structured Visualizations", structured_html))
    
    if show_special_reports:
        print("\nSpecial analysis reports found:")
        for report_name, report_path in show_special_reports:
            print(f"  - {report_name}: {report_path}")
    
    # Open the HTML files
    if os.path.exists(html_path):
        print("\nOpening visualization index in default browser...")
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
            
            # Also open special reports if available
            for report_name, report_path in show_special_reports:
                print(f"Opening {report_name}...")
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Please open these files manually:")
            print(f"  - Visualization index: {html_path}")
            for report_name, report_path in show_special_reports:
                print(f"  - {report_name}: {report_path}")

if __name__ == "__main__":
    main() 