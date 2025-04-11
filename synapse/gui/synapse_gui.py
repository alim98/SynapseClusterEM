#!/usr/bin/env python
"""
Synapse Analysis GUI

A graphical user interface for the synapse analysis pipeline, allowing users to:
- Configure analysis parameters
- Run the analysis pipeline
- View and open generated reports
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import webbrowser
import logging
from datetime import datetime
from PIL import Image, ImageTk
import torch
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SynapseGUI")

class SynapseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Synapse Analysis Tool - Max Planck Institute for Brain Research")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set application icon if available
        try:
            # On Windows, iconbitmap requires .ico format
            if os.path.exists("assets/bioicon.ico"):
                self.root.iconbitmap("assets/bioicon.ico")
            # For cross-platform support using PNG (requires PhotoImage)
            elif os.path.exists("assets/bioicon.png"):
                icon = ImageTk.PhotoImage(Image.open("assets/bioicon.png"))
                self.root.iconphoto(True, icon)
                # Keep a reference to prevent garbage collection
                self.icon_image = icon
        except Exception as e:
            logger.error(f"Error loading icon: {e}")
        
        # Create a style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme
        
        # Create main container
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header with logo
        self.create_header()
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar - initialize before scan_reports is called
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create tabs
        self.setup_config_tab()
        self.setup_visualization_tab()  # Add new visualization tab
        self.setup_run_tab()
        self.setup_reports_tab()
        self.setup_about_tab()
        
        # Initialize variables
        self.running = False
        self.process = None
        self.reports = []
        
        # Scan reports after status_var has been initialized
        self.scan_reports()
        
        # Set up default configurations
        self.load_default_config()
    
    def create_header(self):
        """Create a header with the institute logo"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Use os.path.join for proper path handling
        logo_path = os.path.join("assets", "MPIBRLOGO.png")
        
        # Check if logo exists, otherwise use text
        if not os.path.exists(logo_path):
            logo_label = ttk.Label(
                header_frame, 
                text="Max Planck Institute for Brain Research",
                font=("Arial", 16, "bold")
            )
            logo_label.pack(side=tk.LEFT, padx=10)
            
            # Add instruction label
            instruction_label = ttk.Label(
                header_frame,
                text="(Place 'mpi_logo.png' in the 'assets' folder to display the logo)",
                font=("Arial", 8),
                foreground="gray"
            )
            instruction_label.pack(side=tk.LEFT, padx=5)
        else:
            # Load and display the actual logo
            try:
                # Open the image file
                logo_image = Image.open(logo_path)
                
                # Resize if needed (adjust height to fit in the header)
                max_height = 60
                width, height = logo_image.size
                new_width = int(width * (max_height / height))
                logo_image = logo_image.resize((new_width, max_height), Image.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                tk_image = ImageTk.PhotoImage(logo_image)
                
                # Store reference to prevent garbage collection
                self.logo_image = tk_image
                
                # Create label with image
                logo_label = ttk.Label(header_frame, image=self.logo_image)
                logo_label.pack(side=tk.LEFT, padx=10)
                
                # Add institute name next to logo
                name_label = ttk.Label(
                    header_frame, 
                    text="Max Planck Institute for Brain Research",
                    font=("Arial", 14, "bold")
                )
                name_label.pack(side=tk.LEFT, padx=10)
            except Exception as e:
                # Fallback to text if image loading fails
                logger.error(f"Error loading logo: {e}")
                logo_label = ttk.Label(
                    header_frame, 
                    text="Max Planck Institute for Brain Research",
                    font=("Arial", 16, "bold")
                )
                logo_label.pack(side=tk.LEFT, padx=10)
        
        # Add application title on the right
        title_label = ttk.Label(
            header_frame, 
            text="SynapseClusterEM",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_config_tab(self):
        """Set up the configuration tab"""
        config_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(config_frame, text="Configuration")
        
        # Create a frame for the configuration options with scrollbar
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Data Paths Section
        self.create_section_header(scrollable_frame, "Data Paths")
        
        # Raw data directory
        self.raw_base_dir = self.create_path_entry(
            scrollable_frame, "Raw Data Directory:", "data/7_bboxes_plus_seg/raw"
        )
        
        # Segmentation data directory
        self.seg_base_dir = self.create_path_entry(
            scrollable_frame, "Segmentation Data Directory:", "data/7_bboxes_plus_seg/seg"
        )
        
        # Additional mask directory
        self.add_mask_base_dir = self.create_path_entry(
            scrollable_frame, "Additional Mask Directory:", "data/vesicle_cloud__syn_interface__mitochondria_annotation"
        )
        
        # Excel file
        self.excel_file = self.create_path_entry(
            scrollable_frame, "Excel File:", "data/7_bboxes_plus_seg"
        )
        
        # Output Directories Section
        self.create_section_header(scrollable_frame, "Output Directories")
        
        # CSV output directory
        self.csv_output_dir = self.create_path_entry(
            scrollable_frame, "CSV Output Directory:", "results/csv_outputs"
        )
        
        # GIF output directory
        self.save_gifs_dir = self.create_path_entry(
            scrollable_frame, "GIF Output Directory:", "results/gifs"
        )
        
        # Clustering output directory
        self.clustering_output_dir = self.create_path_entry(
            scrollable_frame, "Clustering Output Directory:", "results/clustering_results_final"
        )
        
        # Report output directory
        self.report_output_dir = self.create_path_entry(
            scrollable_frame, "Report Output Directory:", "results/comprehensive_reports"
        )
        
        # Analysis Parameters Section
        self.create_section_header(scrollable_frame, "Analysis Parameters")
        
        # Bounding box selection
        bbox_frame = ttk.Frame(scrollable_frame)
        bbox_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bbox_frame, text="Bounding Boxes:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.bbox_vars = {}
        bbox_names = ["bbox1", "bbox2", "bbox3", "bbox4", "bbox5", "bbox6", "bbox7"]
        
        for bbox in bbox_names:
            var = tk.BooleanVar(value=(bbox == "bbox1"))  # Default: only bbox1 selected
            self.bbox_vars[bbox] = var
            ttk.Checkbutton(bbox_frame, text=bbox, variable=var).pack(side=tk.LEFT, padx=5)
        
        # Size parameters
        params_frame = ttk.Frame(scrollable_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Size
        ttk.Label(params_frame, text="Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.size_var = tk.IntVar(value=80)
        ttk.Spinbox(params_frame, from_=10, to=200, textvariable=self.size_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Subvolume size
        ttk.Label(params_frame, text="Subvolume Size:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.subvol_size_var = tk.IntVar(value=80)
        ttk.Spinbox(params_frame, from_=10, to=200, textvariable=self.subvol_size_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Number of frames
        ttk.Label(params_frame, text="Number of Frames:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.num_frames_var = tk.IntVar(value=80)
        ttk.Spinbox(params_frame, from_=10, to=200, textvariable=self.num_frames_var, width=5).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Clustering Parameters Section
        self.create_section_header(scrollable_frame, "Clustering Parameters")
        
        clustering_frame = ttk.Frame(scrollable_frame)
        clustering_frame.pack(fill=tk.X, pady=5)
        
        # Clustering algorithm selection
        ttk.Label(clustering_frame, text="Clustering Algorithm:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.clustering_algorithm_var = tk.StringVar(value="KMeans")
        algorithm_combo = ttk.Combobox(clustering_frame, textvariable=self.clustering_algorithm_var, width=10, state="readonly")
        algorithm_combo['values'] = ["KMeans", "DBSCAN"]
        algorithm_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Number of clusters (for KMeans)
        ttk.Label(clustering_frame, text="Number of Clusters:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.n_clusters_var = tk.IntVar(value=10)
        n_clusters_spin = ttk.Spinbox(clustering_frame, from_=2, to=50, textvariable=self.n_clusters_var, width=5)
        n_clusters_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # DBSCAN epsilon parameter
        ttk.Label(clustering_frame, text="DBSCAN Epsilon:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.dbscan_eps_var = tk.DoubleVar(value=0.5)
        dbscan_eps_spin = ttk.Spinbox(clustering_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.dbscan_eps_var, width=5)
        dbscan_eps_spin.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # DBSCAN min_samples parameter
        ttk.Label(clustering_frame, text="DBSCAN Min Samples:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dbscan_min_samples_var = tk.IntVar(value=5)
        dbscan_min_samples_spin = ttk.Spinbox(clustering_frame, from_=1, to=100, textvariable=self.dbscan_min_samples_var, width=5)
        dbscan_min_samples_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Update the enabled/disabled state of parameters based on selected algorithm
        def update_clustering_params(*args):
            if self.clustering_algorithm_var.get() == "KMeans":
                n_clusters_spin.configure(state="normal")
                dbscan_eps_spin.configure(state="disabled")
                dbscan_min_samples_spin.configure(state="disabled")
            else:  # DBSCAN
                n_clusters_spin.configure(state="disabled")
                dbscan_eps_spin.configure(state="normal")
                dbscan_min_samples_spin.configure(state="normal")
        
        # Register the callback
        self.clustering_algorithm_var.trace_add("write", update_clustering_params)
        # Call initially to set the correct states
        update_clustering_params()
        
        # Segmentation Parameters Section
        self.create_section_header(scrollable_frame, "Segmentation Parameters")
        
        seg_frame = ttk.Frame(scrollable_frame)
        seg_frame.pack(fill=tk.X, pady=5)
        
        # Segmentation type
        ttk.Label(seg_frame, text="Segmentation Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.seg_type_var = tk.IntVar(value=10)
        seg_type_combo = ttk.Combobox(seg_frame, textvariable=self.seg_type_var, width=5)
        seg_type_combo['values'] = list(range(0, 11))
        seg_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Alpha value
        ttk.Label(seg_frame, text="Alpha Value:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.alpha_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(seg_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.alpha_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Gray color
        ttk.Label(seg_frame, text="Gray Color:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.gray_color_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(seg_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.gray_color_var, width=5).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Segmentation type description
        seg_desc_frame = ttk.LabelFrame(scrollable_frame, text="Segmentation Type Description", padding=10)
        seg_desc_frame.pack(fill=tk.X, pady=10)
        
        seg_descriptions = {
            0: "Raw data",
            1: "Presynapse",
            2: "Postsynapse",
            3: "Both sides",
            4: "Vesicles + Cleft (closest only)",
            5: "(closest vesicles/cleft + sides)",
            6: "Vesicle cloud (closest)",
            7: "Cleft (closest)",
            8: "Mitochondria (closest)",
            9: "Vesicle + Cleft",
            10: "Cleft + Pre"
        }
        
        for seg_type, desc in seg_descriptions.items():
            ttk.Label(seg_desc_frame, text=f"{seg_type}: {desc}").pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.load_default_config).pack(side=tk.LEFT, padx=5)
    
    def setup_visualization_tab(self):
        """Set up the visualization tab for sample figure creation and preview"""
        try:
            import pandas as pd
            import torch
        except ImportError as e:
            # Handle missing dependencies gracefully
            visualization_frame = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(visualization_frame, text="Visualization")
            error_msg = f"Cannot initialize visualization tab: {str(e)}\n\nPlease ensure all required packages are installed."
            ttk.Label(
                visualization_frame, 
                text=error_msg,
                foreground="red",
                wraplength=700
            ).pack(pady=20)
            logger.error(f"Failed to initialize visualization tab: {str(e)}")
            return
        
        visualization_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(visualization_frame, text="Visualization")
        
        # Create scrollable frame for content
        canvas = tk.Canvas(visualization_frame)
        scrollbar = ttk.Scrollbar(visualization_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create section headers and parameter inputs
        self.create_section_header(scrollable_frame, "Sample Visualization Parameters")
        
        # BBox selection
        bbox_frame = ttk.Frame(scrollable_frame)
        bbox_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bbox_frame, text="Select Bounding Box:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.selected_bbox = tk.StringVar()
        self.bbox_combo = ttk.Combobox(bbox_frame, textvariable=self.selected_bbox, width=10)
        self.bbox_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Sample ID selection
        ttk.Label(bbox_frame, text="Sample ID:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.sample_id_var = tk.StringVar()
        self.sample_id_combo = ttk.Combobox(bbox_frame, textvariable=self.sample_id_var, width=25)
        self.sample_id_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Segmentation parameters
        seg_frame = ttk.Frame(scrollable_frame)
        seg_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seg_frame, text="Segmentation Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.viz_seg_type_var = tk.IntVar(value=5)
        viz_seg_type_combo = ttk.Combobox(seg_frame, textvariable=self.viz_seg_type_var, width=5)
        viz_seg_type_combo['values'] = list(range(0, 11))
        viz_seg_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        # Add trace to reset dataset when parameter changes
        self.viz_seg_type_var.trace_add("write", lambda *args: self.reset_visualization_dataset())
        
        ttk.Label(seg_frame, text="Alpha Value:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.viz_alpha_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(seg_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.viz_alpha_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        # Add trace to reset dataset when parameter changes
        self.viz_alpha_var.trace_add("write", lambda *args: self.reset_visualization_dataset())
        
        ttk.Label(seg_frame, text="Gray Color:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.viz_gray_color_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(seg_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.viz_gray_color_var, width=5).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        # Add trace to reset dataset when parameter changes
        self.viz_gray_color_var.trace_add("write", lambda *args: self.reset_visualization_dataset())
        
        # Output parameters
        output_frame = ttk.Frame(scrollable_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.viz_output_dir = tk.StringVar(value="results/gifs")
        output_entry = ttk.Entry(output_frame, textvariable=self.viz_output_dir, width=40)
        output_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(output_frame, text="Browse...", command=lambda: self.browse_path(self.viz_output_dir)).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Buttons section
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.load_bboxes_btn = ttk.Button(button_frame, text="Load Bounding Boxes", command=self.load_bbox_data)
        self.load_bboxes_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_samples_btn = ttk.Button(button_frame, text="Load Samples", command=self.load_sample_ids, state=tk.DISABLED)
        self.load_samples_btn.pack(side=tk.LEFT, padx=5)
        
        self.visualize_btn = ttk.Button(button_frame, text="Create Visualization", command=self.create_visualization, state=tk.DISABLED)
        self.visualize_btn.pack(side=tk.LEFT, padx=5)
        
        self.open_output_btn = ttk.Button(button_frame, text="Open Output Folder", command=self.open_visualization_folder)
        self.open_output_btn.pack(side=tk.LEFT, padx=5)
        
        # Preview section
        self.preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding=10)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_label = ttk.Label(self.preview_frame, text="No visualization generated yet.\nGenerate a visualization to preview it here.")
        self.preview_label.pack(expand=True, padx=10, pady=10)
        
        # Text output for logs
        log_frame = ttk.LabelFrame(scrollable_frame, text="Visualization Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.viz_log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.viz_log_text.pack(fill=tk.BOTH, expand=True)
        self.viz_log_text.config(state=tk.DISABLED)
        
        # Setup variables for visualization
        self.vol_data_dict = {}
        self.syn_df = None
        self.data_loader = None
        self.processor = None
        self.dataset = None
        self.latest_visualization = None
    
    def create_section_header(self, parent, text):
        """Create a section header"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(10, 5))
        
        header_label = ttk.Label(header_frame, text=text, font=("TkDefaultFont", 10, "bold"))
        header_label.pack(side=tk.LEFT)
        
        separator = ttk.Separator(header_frame, orient=tk.HORIZONTAL)
        separator.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
    
    def create_path_entry(self, parent, label_text, default_value):
        """Create a path entry field with browse button"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(frame, text="Browse", command=lambda: self.browse_path(var)).pack(side=tk.LEFT)
        
        return var
    
    def browse_path(self, var):
        """Open file browser and update path variable"""
        current_path = var.get()
        is_file = os.path.isfile(current_path) if os.path.exists(current_path) else False
        
        if is_file:
            path = filedialog.askopenfilename(initialdir=os.path.dirname(current_path))
        else:
            path = filedialog.askdirectory(initialdir=current_path)
        
        if path:
            var.set(path)
    
    def load_default_config(self):
        """Load default configuration values"""
        # Already set in the constructor
        self.status_var.set("Default configuration loaded.")
    
    def save_config(self):
        """Save the current configuration to a file"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".conf",
            filetypes=[("Configuration files", "*.conf"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        with open(filepath, 'w') as f:
            f.write("[Paths]\n")
            f.write(f"raw_base_dir = {self.raw_base_dir.get()}\n")
            f.write(f"seg_base_dir = {self.seg_base_dir.get()}\n")
            f.write(f"add_mask_base_dir = {self.add_mask_base_dir.get()}\n")
            f.write(f"excel_file = {self.excel_file.get()}\n")
            f.write(f"csv_output_dir = {self.csv_output_dir.get()}\n")
            f.write(f"save_gifs_dir = {self.save_gifs_dir.get()}\n")
            f.write(f"clustering_output_dir = {self.clustering_output_dir.get()}\n")
            f.write(f"report_output_dir = {self.report_output_dir.get()}\n")
            
            f.write("\n[Parameters]\n")
            # Bounding boxes
            bbox_list = [bbox for bbox, var in self.bbox_vars.items() if var.get()]
            f.write(f"bbox_name = {','.join(bbox_list)}\n")
            
            # Size parameters
            f.write(f"size = {self.size_var.get()}\n")
            f.write(f"subvol_size = {self.subvol_size_var.get()}\n")
            f.write(f"num_frames = {self.num_frames_var.get()}\n")
            
            # Clustering parameters
            f.write(f"clustering_algorithm = {self.clustering_algorithm_var.get()}\n")
            f.write(f"n_clusters = {self.n_clusters_var.get()}\n")
            f.write(f"dbscan_eps = {self.dbscan_eps_var.get()}\n")
            f.write(f"dbscan_min_samples = {self.dbscan_min_samples_var.get()}\n")
            
            # Segmentation parameters
            f.write(f"segmentation_type = {self.seg_type_var.get()}\n")
            f.write(f"alpha = {self.alpha_var.get()}\n")
            f.write(f"gray_color = {self.gray_color_var.get()}\n")
        
        messagebox.showinfo("Configuration Saved", f"Configuration saved to {filepath}")
    
    def load_config(self):
        """Load configuration from a file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Configuration files", "*.conf;*.ini"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if not filepath:
            return
        
        try:
            # Parse the config file
            config = {'Paths': {}, 'Parameters': {}}
            current_section = None
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        continue
                    
                    if '=' in line and current_section:
                        key, value = line.split('=', 1)
                        config[current_section][key.strip()] = value.strip()
            
            # Update paths
            if 'Paths' in config:
                if 'raw_base_dir' in config['Paths']:
                    self.raw_base_dir.set(config['Paths']['raw_base_dir'])
                if 'seg_base_dir' in config['Paths']:
                    self.seg_base_dir.set(config['Paths']['seg_base_dir'])
                if 'add_mask_base_dir' in config['Paths']:
                    self.add_mask_base_dir.set(config['Paths']['add_mask_base_dir'])
                if 'excel_file' in config['Paths']:
                    self.excel_file.set(config['Paths']['excel_file'])
                if 'csv_output_dir' in config['Paths']:
                    self.csv_output_dir.set(config['Paths']['csv_output_dir'])
                if 'save_gifs_dir' in config['Paths']:
                    self.save_gifs_dir.set(config['Paths']['save_gifs_dir'])
                if 'clustering_output_dir' in config['Paths']:
                    self.clustering_output_dir.set(config['Paths']['clustering_output_dir'])
                if 'report_output_dir' in config['Paths']:
                    self.report_output_dir.set(config['Paths']['report_output_dir'])
            
            # Update parameters
            if 'Parameters' in config:
                if 'bbox_name' in config['Parameters']:
                    bbox_list = config['Parameters']['bbox_name'].split(',')
                    for bbox, var in self.bbox_vars.items():
                        var.set(bbox in bbox_list)
                
                if 'size' in config['Parameters']:
                    self.size_var.set(int(config['Parameters']['size']))
                if 'subvol_size' in config['Parameters']:
                    self.subvol_size_var.set(int(config['Parameters']['subvol_size']))
                if 'num_frames' in config['Parameters']:
                    self.num_frames_var.set(int(config['Parameters']['num_frames']))
                
                # Load clustering parameters
                if 'clustering_algorithm' in config['Parameters']:
                    self.clustering_algorithm_var.set(config['Parameters']['clustering_algorithm'])
                if 'n_clusters' in config['Parameters']:
                    self.n_clusters_var.set(int(config['Parameters']['n_clusters']))
                if 'dbscan_eps' in config['Parameters']:
                    self.dbscan_eps_var.set(float(config['Parameters']['dbscan_eps']))
                if 'dbscan_min_samples' in config['Parameters']:
                    self.dbscan_min_samples_var.set(int(config['Parameters']['dbscan_min_samples']))
                
                if 'segmentation_type' in config['Parameters']:
                    self.seg_type_var.set(int(config['Parameters']['segmentation_type']))
                if 'alpha' in config['Parameters']:
                    self.alpha_var.set(float(config['Parameters']['alpha']))
                if 'gray_color' in config['Parameters']:
                    self.gray_color_var.set(float(config['Parameters']['gray_color']))
            
            self.status_var.set(f"Configuration loaded from {filepath}")
            logger.info(f"Configuration loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
            logger.error(f"Failed to load configuration: {str(e)}")
    
    def run_analysis(self):
        """Run the synapse analysis with current settings"""
        if self.running:
            messagebox.showinfo("Already Running", "Analysis is already running.")
            return
        
        # Build command
        cmd = ["python", "inference.py"]
        
        # Add path arguments
        cmd.extend(["--raw_base_dir", self.raw_base_dir.get()])
        cmd.extend(["--seg_base_dir", self.seg_base_dir.get()])
        cmd.extend(["--add_mask_base_dir", self.add_mask_base_dir.get()])
        cmd.extend(["--excel_file", self.excel_file.get()])
        cmd.extend(["--csv_output_dir", self.csv_output_dir.get()])
        cmd.extend(["--save_gifs_dir", self.save_gifs_dir.get()])
        cmd.extend(["--clustering_output_dir", self.clustering_output_dir.get()])
        cmd.extend(["--report_output_dir", self.report_output_dir.get()])
        
        # Add bounding boxes
        bbox_list = [bbox for bbox, var in self.bbox_vars.items() if var.get()]
        if bbox_list:
            cmd.extend(["--bbox_name"] + bbox_list)
        
        # Add size parameters
        cmd.extend(["--subvol_size", str(self.subvol_size_var.get())])
        cmd.extend(["--num_frames", str(self.num_frames_var.get())])
        
        # Add segmentation parameters
        cmd.extend(["--segmentation_type", str(self.seg_type_var.get())])
        cmd.extend(["--alpha", str(self.alpha_var.get())])
        cmd.extend(["--gray_color", str(self.gray_color_var.get())])
        
        # Add clustering parameters
        cmd.extend(["--clustering_algorithm", self.clustering_algorithm_var.get()])
        
        if self.clustering_algorithm_var.get() == "KMeans":
            cmd.extend(["--n_clusters", str(self.n_clusters_var.get())])
        else:  # DBSCAN
            cmd.extend(["--dbscan_eps", str(self.dbscan_eps_var.get())])
            cmd.extend(["--dbscan_min_samples", str(self.dbscan_min_samples_var.get())])
        
        # Add analysis flags
        if not self.feature_extraction_var.get():
            cmd.append("--skip_feature_extraction")
        
        if not self.clustering_var.get():
            cmd.append("--skip_clustering")
        
        if not self.presynapse_analysis_var.get():
            cmd.append("--skip_presynapse_analysis")
        
        if not self.report_generation_var.get():
            cmd.append("--skip_report_generation")
            
        # Add only feature extraction and clustering flag if selected
        if self.only_feature_clustering_var.get():
            cmd.append("--only_feature_extraction_and_clustering")
        
        # Update UI
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Analysis running...")
        
        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Run in a separate thread
        logger.info("Starting analysis with command: " + " ".join(cmd))
        
        def run_process():
            try:
                # Run the process
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Read output
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        logger.info(line.strip())
                
                # Wait for process to complete
                self.process.wait()
                
                # Update UI
                self.root.after(0, self.process_complete)
            except Exception as e:
                logger.error(f"Error running analysis: {str(e)}")
                self.root.after(0, self.process_error, str(e))
        
        threading.Thread(target=run_process, daemon=True).start()
    
    def process_complete(self):
        """Called when the analysis process is complete"""
        self.running = False
        self.process = None
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.report_generation_var.get():
            self.scan_reports()
            self.notebook.select(2)  # Switch to Reports tab
        
        self.status_var.set("Analysis complete")
        logger.info("Analysis completed successfully")
        messagebox.showinfo("Complete", "Synapse analysis completed successfully!")
    
    def process_error(self, error_msg):
        """Called when there's an error in the analysis process"""
        self.running = False
        self.process = None
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Analysis failed")
        
        messagebox.showerror("Error", f"Analysis failed: {error_msg}")
    
    def stop_analysis(self):
        """Stop the running analysis process"""
        if not self.running or self.process is None:
            return
        
        try:
            self.process.terminate()
            logger.info("Process terminated by user")
            self.status_var.set("Analysis stopped by user")
            
            self.running = False
            self.process = None
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error stopping process: {str(e)}")
    
    def scan_reports(self):
        """Scan for available reports"""
        report_dir = Path(self.report_output_dir.get())
        if not report_dir.exists():
            return
        
        # Clear existing items
        for item in self.reports_tree.get_children():
            self.reports_tree.delete(item)
        
        # Find comprehensive reports
        self.reports = []
        
        # Find comprehensive reports (regular format)
        comp_reports = list(report_dir.glob("report_*"))
        for report_path in comp_reports:
            if report_path.is_dir():
                # Extract date from report name
                date_str = report_path.name.replace("report_", "")
                try:
                    date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_date = date_str
                
                index = len(self.reports)
                report_item = {
                    "type": "Comprehensive Report",
                    "date": formatted_date,
                    "path": str(report_path / "index.html"),
                    "dir": str(report_path)
                }
                self.reports.append(report_item)
                
                self.reports_tree.insert("", tk.END, iid=str(index), 
                                         values=(report_item["type"], report_item["date"], report_item["path"]))
        
        # Find presynapse summary reports
        pre_reports = list(report_dir.glob("presynapse_summary_*"))
        for report_path in pre_reports:
            if report_path.is_dir():
                # Extract date from report name
                date_str = report_path.name.replace("presynapse_summary_", "")
                try:
                    date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_date = date_str
                
                index = len(self.reports)
                report_item = {
                    "type": "Presynapse Summary",
                    "date": formatted_date,
                    "path": str(report_path / "presynapse_summary.html"),
                    "dir": str(report_path)
                }
                self.reports.append(report_item)
                
                self.reports_tree.insert("", tk.END, iid=str(index), 
                                         values=(report_item["type"], report_item["date"], report_item["path"]))
        
        # Update status
        self.status_var.set(f"Found {len(self.reports)} reports")
    
    def open_selected_report(self, event):
        """Open the selected report when double-clicked"""
        self.open_report()
    
    def open_report(self):
        """Open the selected report in the default browser"""
        selection = self.reports_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a report to open.")
            return
        
        index = int(selection[0])
        report_path = self.reports[index]["path"]
        
        try:
            # Convert to absolute path if needed
            abs_path = os.path.abspath(report_path)
            webbrowser.open(f"file://{abs_path}")
            logger.info(f"Opened report: {abs_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {str(e)}")
            logger.error(f"Failed to open report: {str(e)}")
    
    def open_report_folder(self):
        """Open the folder containing the selected report"""
        selection = self.reports_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a report to open its folder.")
            return
        
        index = int(selection[0])
        report_dir = self.reports[index]["dir"]
        
        try:
            # Convert to absolute path if needed
            abs_path = os.path.abspath(report_dir)
            if sys.platform == 'win32':
                os.startfile(abs_path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', abs_path])
            else:
                subprocess.call(['xdg-open', abs_path])
            
            logger.info(f"Opened report directory: {abs_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
            logger.error(f"Failed to open folder: {str(e)}")
    
    def delete_report(self):
        """Delete the selected report"""
        selection = self.reports_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a report to delete.")
            return
        
        index = int(selection[0])
        report_type = self.reports[index]["type"]
        report_dir = self.reports[index]["dir"]
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the {report_type}?"):
            try:
                import shutil
                shutil.rmtree(report_dir)
                logger.info(f"Deleted report directory: {report_dir}")
                
                # Refresh reports list
                self.scan_reports()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete report: {str(e)}")
                logger.error(f"Failed to delete report: {str(e)}")
    
    def load_bbox_data(self):
        """Load bounding box data for visualization"""
        from synapse import SynapseDataLoader, config
        import pandas as pd
        
        # Update status
        self.status_var.set("Loading bounding box data...")
        self.root.update_idletasks()
        
        # Get config parameters from the UI
        raw_base_dir = self.raw_base_dir.get()
        seg_base_dir = self.seg_base_dir.get()
        add_mask_base_dir = self.add_mask_base_dir.get()
        excel_file = self.excel_file.get()
        
        # Check if paths exist
        if not os.path.exists(raw_base_dir) or not os.path.exists(seg_base_dir) or not os.path.exists(add_mask_base_dir):
            messagebox.showerror("Error", "One or more data directories do not exist. Please check your paths.")
            self.status_var.set("Error loading bounding box data")
            return
        
        try:
            self.log_to_viz_console("Loading data loader...")
            self.data_loader = SynapseDataLoader(
                raw_base_dir=raw_base_dir,
                seg_base_dir=seg_base_dir,
                add_mask_base_dir=add_mask_base_dir,
                gray_color=self.viz_gray_color_var.get()
            )
            
            # Get potential bboxes from directories
            possible_bboxes = []
            for item in os.listdir(raw_base_dir):
                if os.path.isdir(os.path.join(raw_base_dir, item)) and "bbox" in item.lower():
                    possible_bboxes.append(item)
            
            if not possible_bboxes:
                messagebox.showerror("Error", "No bounding box directories found in the raw data directory.")
                self.status_var.set("No bounding boxes found")
                return
            
            # Update the combo box
            self.bbox_combo['values'] = possible_bboxes
            self.bbox_combo.current(0)  # Set the first bbox as default
            
            # Enable the load samples button
            self.load_samples_btn.config(state=tk.NORMAL)
            
            self.log_to_viz_console(f"Found {len(possible_bboxes)} bounding boxes: {', '.join(possible_bboxes)}")
            self.status_var.set(f"Loaded {len(possible_bboxes)} bounding boxes")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading bounding box data: {str(e)}")
            self.status_var.set("Error loading bounding box data")
            self.log_to_viz_console(f"Error: {str(e)}")
    
    def load_sample_ids(self):
        """Load sample IDs for the selected bounding box"""
        import pandas as pd
        from synapse import config
        
        # Update status
        self.status_var.set("Loading sample IDs...")
        self.root.update_idletasks()
        
        bbox_name = self.selected_bbox.get()
        excel_file = self.excel_file.get()
        
        if not bbox_name:
            messagebox.showerror("Error", "Please select a bounding box first.")
            return
        
        try:
            # Load the volume data if not already loaded
            if bbox_name not in self.vol_data_dict:
                self.log_to_viz_console(f"Loading volumes for {bbox_name}...")
                raw_vol, seg_vol, add_mask_vol = self.data_loader.load_volumes(bbox_name)
                
                if raw_vol is None:
                    messagebox.showerror("Error", f"Could not load volumes for {bbox_name}.")
                    self.status_var.set(f"Error loading volumes for {bbox_name}")
                    return
                
                self.vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
                self.log_to_viz_console(f"Loaded volumes for {bbox_name} with shape {raw_vol.shape}")
            
            # Load the excel file for the bbox
            excel_path = os.path.join(excel_file, f"{bbox_name}.xlsx")
            if not os.path.exists(excel_path):
                messagebox.showerror("Error", f"Excel file for {bbox_name} not found at {excel_path}")
                self.status_var.set(f"Excel file for {bbox_name} not found")
                return
            
            self.log_to_viz_console(f"Loading Excel data from {excel_path}...")
            # Load only the relevant bbox data
            df = pd.read_excel(excel_path)
            df['bbox_name'] = bbox_name
            
            # Store the dataframe
            self.syn_df = df
            
            # Get Var1 values for the sample IDs
            if 'Var1' in df.columns:
                sample_ids = df['Var1'].unique().tolist()
                self.sample_id_combo['values'] = sample_ids
                if sample_ids:
                    self.sample_id_combo.current(0)  # Set the first sample as default
                    self.visualize_btn.config(state=tk.NORMAL)
                    self.log_to_viz_console(f"Loaded {len(sample_ids)} sample IDs")
                    self.status_var.set(f"Loaded {len(sample_ids)} sample IDs for {bbox_name}")
                else:
                    self.log_to_viz_console("No sample IDs found in the Excel file")
                    self.status_var.set("No sample IDs found")
            else:
                messagebox.showwarning("Warning", "The Excel file does not contain a 'Var1' column for sample IDs.")
                self.status_var.set("No Var1 column in Excel file")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading sample IDs: {str(e)}")
            self.status_var.set("Error loading sample IDs")
            self.log_to_viz_console(f"Error: {str(e)}")
    
    def create_visualization(self):
        """Create a visualization for the selected sample"""
        
        # Import from newdl module instead of synapse
        from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
        from newdl.dataset3 import SynapseDataset
        from synapse import  config
        import numpy as np
        from PIL import Image, ImageTk
        import matplotlib.pyplot as plt
        
        # Try to import visualization utilities
        try:
            from synapse.visualization.sample_fig import save_center_slice_image
        except ImportError:
            # Fallback implementation if the import fails
            def save_center_slice_image(volume, output_path, consistent_gray=True):
                """Local fallback implementation if the import fails"""
                self.log_to_viz_console("Using fallback save_center_slice_image implementation")
                
                # Get center slice
                if len(volume.shape) == 4:  # (z, c, y, x)
                    center_slice = volume[volume.shape[0] // 2, 0]
                else:  # (z, y, x)
                    center_slice = volume[volume.shape[0] // 2]
                
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create the figure with controlled normalization
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
                if consistent_gray:
                    ax.imshow(center_slice, cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(center_slice, cmap='gray')
                
                ax.axis('off')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
                plt.close()
                
                self.log_to_viz_console(f"Center slice image saved to {output_path}")
        
        # Update status
        self.status_var.set("Creating visualization...")
        self.root.update_idletasks()
        
        bbox_name = self.selected_bbox.get()
        sample_id = self.sample_id_var.get()
        segmentation_type = self.viz_seg_type_var.get()
        alpha = self.viz_alpha_var.get()
        gray_color = self.viz_gray_color_var.get()
        output_dir = self.viz_output_dir.get()
        
        if not (bbox_name and sample_id):
            messagebox.showerror("Error", "Please select both a bounding box and a sample ID.")
            return
        
        try:
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Set the gray color for visualization
            self.data_loader.gray_color = gray_color
            
            # Initialize processor if not already done
            if self.processor is None:
                self.processor = Synapse3DProcessor(size=config.size)
                self.log_to_viz_console("Initialized processor")
            
            # Create or update dataset with current parameters
            self.log_to_viz_console(f"Creating dataset with segmentation type {segmentation_type}, alpha {alpha}")
            
            try:
                self.dataset = SynapseDataset(
                    vol_data_dict=self.vol_data_dict,
                    synapse_df=self.syn_df,
                    processor=self.processor,
                    segmentation_type=segmentation_type,
                    subvol_size=self.subvol_size_var.get(),
                    num_frames=self.num_frames_var.get(),
                    alpha=alpha
                )
                self.log_to_viz_console(f"Created dataset with {len(self.dataset)} samples")
            except Exception as dataset_error:
                messagebox.showerror("Dataset Error", f"Failed to create dataset: {str(dataset_error)}")
                self.log_to_viz_console(f"Dataset error: {str(dataset_error)}")
                self.status_var.set("Error creating dataset")
                return
            
            # Find the sample in the synapse dataframe
            sample_df = self.syn_df[(self.syn_df['Var1'] == sample_id) & (self.syn_df['bbox_name'] == bbox_name)]
            
            if sample_df.empty:
                messagebox.showerror("Error", f"Sample {sample_id} not found in {bbox_name}.")
                return
            
            # Get the sample data
            idx = sample_df.index[0]
            pixel_values, syn_info, _ = self.dataset[idx]
            
            # Convert to numpy array for visualization
            if isinstance(pixel_values, torch.Tensor):
                frames = pixel_values.squeeze(1).numpy()
            else:
                frames = pixel_values.squeeze(1)
            
            # Prepare the output path
            Gif_Name = f"{bbox_name}_{sample_id}_seg{segmentation_type}"
            output_gif_path = os.path.join(output_dir, f"{Gif_Name}.gif")
            output_png_path = os.path.join(output_dir, f"{Gif_Name}_center_slice.png")
            
            # Save the center slice as a PNG
            self.log_to_viz_console(f"Saving center slice to {output_png_path}...")
            center_slice_idx = frames.shape[0] // 2
            center_slice = frames[center_slice_idx]
            
            # Save using the utility function that preserves gray levels
            save_center_slice_image(frames, output_png_path, consistent_gray=True)
            
            # Create and save the GIF
            self.log_to_viz_console(f"Creating GIF with {len(frames)} frames...")
            enhanced_frames = []
            for frame in frames:
                # Min-max normalization to stretch contrast
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max > frame_min:  # Avoid division by zero
                    normalized = (frame - frame_min) / (frame_max - frame_min)
                else:
                    normalized = frame
                enhanced_frames.append((normalized * 255).astype(np.uint8))
            
            import imageio
            imageio.mimsave(output_gif_path, enhanced_frames, fps=10)
            
            # Show the center slice in the preview
            self.log_to_viz_console("Loading preview image...")
            self.latest_visualization = output_png_path
            self.update_preview(output_png_path)
            
            # Update status
            self.status_var.set(f"Visualization created and saved to {output_dir}")
            self.log_to_viz_console(f"Visualization successfully created and saved to {output_dir}")
            
            # Enable the open output folder button
            self.open_output_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the visualization: {str(e)}")
            self.status_var.set("Error creating visualization")
            self.log_to_viz_console(f"Error: {str(e)}")
    
    def update_preview(self, image_path):
        """Update the preview with the generated visualization"""
        from PIL import Image, ImageTk
        
        # Clear the current preview
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        try:
            # Open and resize the image for preview
            img = Image.open(image_path)
            
            # Calculate the resize ratio to fit the preview area
            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            tk_img = ImageTk.PhotoImage(img)
            
            # Display the image
            img_label = ttk.Label(self.preview_frame, image=tk_img)
            img_label.image = tk_img  # Keep a reference to prevent garbage collection
            img_label.pack(padx=10, pady=10)
            
            # Add a label with the image info
            ttk.Label(
                self.preview_frame, 
                text=f"Preview of: {os.path.basename(image_path)}"
            ).pack(padx=10, pady=5)
            
        except Exception as e:
            ttk.Label(
                self.preview_frame, 
                text=f"Error loading preview: {str(e)}"
            ).pack(padx=10, pady=10)
    
    def open_visualization_folder(self):
        """Open the folder containing the visualizations"""
        output_dir = self.viz_output_dir.get()
        
        # Create the directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.log_to_viz_console(f"Created output directory: {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create directory: {str(e)}")
            self.status_var.set(f"Error creating directory: {str(e)}")
            return
        
        try:
            # Convert to absolute path with proper separators for the OS
            abs_path = os.path.abspath(output_dir)
            
            # Open the folder using the default file explorer
            if sys.platform == 'win32':
                # On Windows, ensure the path uses backslashes
                abs_path = abs_path.replace('/', '\\')
                os.startfile(abs_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', abs_path])
            else:  # Linux
                subprocess.run(['xdg-open', abs_path])
                
            self.status_var.set(f"Opened visualization folder: {abs_path}")
            self.log_to_viz_console(f"Opened folder: {abs_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
            self.status_var.set(f"Error opening folder: {str(e)}")
            self.log_to_viz_console(f"Error: {str(e)}")
    
    def log_to_viz_console(self, message):
        """Log a message to the visualization console"""
        self.viz_log_text.config(state=tk.NORMAL)
        self.viz_log_text.insert(tk.END, f"{message}\n")
        self.viz_log_text.see(tk.END)  # Auto-scroll to the end
        self.viz_log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()  # Update the UI
    
    def reset_visualization_dataset(self):
        """Reset the visualization dataset when parameters change"""
        self.dataset = None
        self.log_to_viz_console("Visualization parameters changed - dataset will be recreated")
    
    def setup_run_tab(self):
        """Set up the run tab for executing the analysis"""
        run_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(run_frame, text="Run Analysis")
        
        # Create top frame for options
        options_frame = ttk.LabelFrame(run_frame, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Checkboxes for different parts of the analysis
        self.feature_extraction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Feature Extraction", variable=self.feature_extraction_var).grid(row=0, column=0, sticky=tk.W, padx=10)
        
        self.clustering_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Clustering Analysis", variable=self.clustering_var).grid(row=0, column=1, sticky=tk.W, padx=10)
        
        self.presynapse_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Presynapse Analysis", variable=self.presynapse_analysis_var).grid(row=0, column=2, sticky=tk.W, padx=10)
        
        self.report_generation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Report Generation", variable=self.report_generation_var).grid(row=0, column=3, sticky=tk.W, padx=10)
        
        # Add a separator
        ttk.Separator(options_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', pady=5)
        
        # Add "Only Feature Extraction and Clustering" option
        self.only_feature_clustering_var = tk.BooleanVar(value=False)
        only_feature_clustering_cb = ttk.Checkbutton(
            options_frame, 
            text="Only Feature Extraction and Clustering", 
            variable=self.only_feature_clustering_var
        )
        only_feature_clustering_cb.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=10)
        
        # Add trace to disable other options when "Only Feature Extraction and Clustering" is selected
        def update_analysis_options(*args):
            if self.only_feature_clustering_var.get():
                # If "Only Feature Extraction and Clustering" is selected, disable other options
                self.presynapse_analysis_var.set(False)
                self.report_generation_var.set(False)
            
        # Register the callback
        self.only_feature_clustering_var.trace_add("write", update_analysis_options)
        
        # Create buttons frame
        button_frame = ttk.Frame(run_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the log display
        log_frame = ttk.LabelFrame(run_frame, text="Analysis Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Text widget for log output
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Add custom handler to redirect log output to text widget
        self.log_handler = TextHandler(self.log_text)
        self.log_handler.setLevel(logging.INFO)
        logger.addHandler(self.log_handler)
    
    def setup_reports_tab(self):
        """Set up the reports tab for viewing generated reports"""
        reports_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(reports_frame, text="Reports")
        
        # Create refresh button
        refresh_frame = ttk.Frame(reports_frame)
        refresh_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(refresh_frame, text="Refresh Reports List", command=self.scan_reports).pack(side=tk.LEFT)
        
        # Create the reports list frame
        list_frame = ttk.LabelFrame(reports_frame, text="Available Reports", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a frame with a tree view for reports
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for reports
        self.reports_tree = ttk.Treeview(tree_frame, columns=("Type", "Date", "Path"), show="headings")
        self.reports_tree.heading("Type", text="Report Type")
        self.reports_tree.heading("Date", text="Date")
        self.reports_tree.heading("Path", text="Path")
        
        self.reports_tree.column("Type", width=150)
        self.reports_tree.column("Date", width=150)
        self.reports_tree.column("Path", width=400)
        
        self.reports_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.reports_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.reports_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind double-click to open report
        self.reports_tree.bind("<Double-1>", self.open_selected_report)
        
        # Create buttons for report actions
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Open Report", command=self.open_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open Report Folder", command=self.open_report_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Report", command=self.delete_report).pack(side=tk.LEFT, padx=5)
    
    def setup_about_tab(self):
        """Set up the about tab with information about the project and institute"""
        about_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(about_frame, text="About")
        
        # Project title
        title_label = ttk.Label(
            about_frame, 
            text="SynapseClusterEM",
            font=("Arial", 18, "bold")
        )
        title_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Version
        version_label = ttk.Label(
            about_frame,
            text="Version 1.0.0",
            font=("Arial", 10)
        )
        version_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Description
        description = (
            "SynapseClusterEM is a deep learning framework for analyzing and clustering "
            "3D synapse structures from electron microscopy (EM) data. The tool uses "
            "advanced neural networks and unsupervised learning techniques to identify "
            "structural patterns and classify synapses based on their 3D architecture."
        )
        desc_label = ttk.Label(
            about_frame,
            text=description,
            wraplength=700,
            justify=tk.LEFT
        )
        desc_label.pack(anchor=tk.W, pady=(0, 20), fill=tk.X)
        
        # Institute information
        institute_frame = ttk.LabelFrame(about_frame, text="Developed at", padding=10)
        institute_frame.pack(fill=tk.X, pady=10)
        
        institute_name = ttk.Label(
            institute_frame,
            text="Max Planck Institute for Brain Research",
            font=("Arial", 12, "bold")
        )
        institute_name.pack(anchor=tk.W)
        
        institute_address = ttk.Label(
            institute_frame,
            text="Frankfurt am Main, Germany",
            font=("Arial", 10)
        )
        institute_address.pack(anchor=tk.W)
        
        institute_website = ttk.Label(
            institute_frame,
            text="https://brain.mpg.de/",
            font=("Arial", 10),
            foreground="blue",
            cursor="hand2"
        )
        institute_website.pack(anchor=tk.W, pady=(0, 10))
        institute_website.bind("<Button-1>", lambda e: webbrowser.open_new("https://brain.mpg.de/"))
        
        # Developer information
        developer_frame = ttk.LabelFrame(about_frame, text="Development Team", padding=10)
        developer_frame.pack(fill=tk.X, pady=10)
        
        developer_info = ttk.Label(
            developer_frame,
            text="Ali Mikaeili",
            font=("Arial", 10, "bold")
        )
        developer_info.pack(anchor=tk.W)
        
        contact_info = ttk.Label(
            developer_frame,
            text="Email: Mikaeili.Barzili@gmail.com",
            font=("Arial", 10)
        )
        contact_info.pack(anchor=tk.W)
        
        github_link = ttk.Label(
            developer_frame,
            text="GitHub: https://github.com/alim98",
            font=("Arial", 10),
            foreground="blue",
            cursor="hand2"
        )
        github_link.pack(anchor=tk.W)
        
        developer_info2 = ttk.Label(
            developer_frame,
            text="Ali Karimi - Postdoc",
            font=("Arial", 10, "bold")
        )
        developer_info2.pack(anchor=tk.W, pady=(10, 0))
        
        developer_info3 = ttk.Label(
            developer_frame,
            text="Dominic Evans - Postdoc",
            font=("Arial", 10, "bold")
        )
        developer_info3.pack(anchor=tk.W, pady=(10, 0))
        github_link.pack(anchor=tk.W)
        github_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/alim98"))
        
        # Copyright notice
        copyright_label = ttk.Label(
            about_frame,
            text="© 2025 Max Planck Institute for Brain Research. All rights reserved.",
            font=("Arial", 9),
            foreground="gray"
        )
        copyright_label.pack(side=tk.BOTTOM, pady=20)


class TextHandler(logging.Handler):
    """Handler that redirects logging output to a tkinter Text widget"""
    
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        
        # Configure format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)
    
    def emit(self, record):
        msg = self.format(record)
        
        def append():
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
        
        # Ensure thread safety by using the after method to schedule GUI updates
        self.text_widget.after(0, append)


if __name__ == "__main__":
    root = tk.Tk()
    app = SynapseGUI(root)
    root.mainloop() 