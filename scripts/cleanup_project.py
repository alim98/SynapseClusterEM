#!/usr/bin/env python
"""
Project Cleanup Script

This script removes unnecessary files from the project directory:
1. macOS metadata files (._*)
2. .DS_Store files
3. Other temporary files

Usage:
    python scripts/cleanup_project.py [--dry-run]
"""

import os
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean up unnecessary files from the project")
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    return parser.parse_args()


def cleanup_directory(directory, patterns_to_remove, dry_run=False):
    """
    Clean up a directory by removing files matching the specified patterns.
    
    Args:
        directory: Directory to clean
        patterns_to_remove: List of file patterns to remove
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Tuple of (number of files removed, total size freed)
    """
    files_removed = 0
    size_freed = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file matches any pattern to remove
            should_remove = any(
                (pattern.startswith('*') and file.endswith(pattern[1:])) or
                (pattern.endswith('*') and file.startswith(pattern[:-1])) or
                (pattern == file)
                for pattern in patterns_to_remove
            )
            
            if should_remove:
                file_size = os.path.getsize(file_path)
                if dry_run:
                    print(f"Would remove: {file_path} ({file_size} bytes)")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path} ({file_size} bytes)")
                        files_removed += 1
                        size_freed += file_size
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
    
    return files_removed, size_freed


def main():
    args = parse_args()
    
    # Get project root directory (parent of the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Patterns of files to remove
    patterns_to_remove = [
        "._*",        # macOS metadata files
        ".DS_Store",  # macOS folder metadata
        "*.pyc",      # Python compiled files
        "__pycache__" # Python cache directories
    ]
    
    print(f"{'Dry run: ' if args.dry_run else ''}Cleaning up project directory: {project_root}")
    
    # Clean up the project directory
    files_removed, size_freed = cleanup_directory(
        project_root, 
        patterns_to_remove, 
        dry_run=args.dry_run
    )
    
    # Format size in a human-readable format
    if size_freed < 1024:
        size_str = f"{size_freed} bytes"
    elif size_freed < 1024 * 1024:
        size_str = f"{size_freed / 1024:.2f} KB"
    else:
        size_str = f"{size_freed / (1024 * 1024):.2f} MB"
    
    if args.dry_run:
        print(f"\nDry run complete. Would remove {files_removed} files, freeing {size_str}.")
        print("Run without --dry-run to actually delete these files.")
    else:
        print(f"\nCleanup complete. Removed {files_removed} files, freeing {size_str}.")


if __name__ == "__main__":
    main() 