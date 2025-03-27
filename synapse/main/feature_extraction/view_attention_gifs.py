"""
Attention GIF Viewer

A simple utility to view and browse the attention map GIFs.
"""

import os
import sys
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageSequence
import argparse
from pathlib import Path

class GifViewer:
    def __init__(self, root, gif_dir):
        self.root = root
        self.root.title("Attention GIF Viewer")
        
        # Get all GIF files in the directory and its subdirectories
        self.gif_files = []
        for dir_path, _, _ in os.walk(gif_dir):
            self.gif_files.extend(glob.glob(os.path.join(dir_path, "*.gif")))
        
        # Sort the files by segmentation type and sample number
        self.gif_files.sort()
        
        if not self.gif_files:
            print(f"No GIF files found in {gif_dir}")
            sys.exit(1)
            
        # Setup UI
        self.current_index = 0
        self.frame_index = 0
        self.playing = False
        self.delay = 100  # ms between frames
        
        # Create frames
        self.top_frame = ttk.Frame(root)
        self.top_frame.pack(fill="x", pady=5)
        
        self.image_frame = ttk.Frame(root)
        self.image_frame.pack(fill="both", expand=True, pady=5)
        
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill="x", pady=5)
        
        # Filename and info display
        self.info_label = ttk.Label(self.top_frame, text="")
        self.info_label.pack(side="left", padx=10)
        
        # File navigation
        file_nav_frame = ttk.Frame(self.top_frame)
        file_nav_frame.pack(side="right", padx=10)
        
        ttk.Button(file_nav_frame, text="Prev File", command=self.prev_file).pack(side="left", padx=5)
        ttk.Button(file_nav_frame, text="Next File", command=self.next_file).pack(side="left", padx=5)
        
        # Image display
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill="both", expand=True)
        
        # Playback controls
        ttk.Button(self.control_frame, text="Play/Pause", command=self.toggle_play).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Prev Frame", command=self.prev_frame).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Next Frame", command=self.next_frame).pack(side="left", padx=5)
        
        # Speed control
        ttk.Label(self.control_frame, text="Speed:").pack(side="left", padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(self.control_frame, from_=0.1, to=3.0, 
                               variable=self.speed_var, orient="horizontal",
                               length=100, command=self.update_speed)
        speed_scale.pack(side="left", padx=5)
        
        # Load first GIF
        self.load_current_gif()
        
        # Key bindings
        self.root.bind("<Left>", lambda e: self.prev_file())
        self.root.bind("<Right>", lambda e: self.next_file())
        self.root.bind("<space>", lambda e: self.toggle_play())
        
    def load_current_gif(self):
        """Load the current GIF file"""
        self.playing = False
        
        # Get file path and extract info
        gif_path = self.gif_files[self.current_index]
        file_name = os.path.basename(gif_path)
        seg_type = "unknown"
        
        # Try to extract segmentation type from the path
        if "seg_type_" in gif_path:
            parts = gif_path.split("seg_type_")
            if len(parts) > 1:
                seg_type = parts[1].split(os.path.sep)[0]
        
        # Prepare descriptive label
        self.info_label.config(text=f"File {self.current_index+1}/{len(self.gif_files)}: {file_name}\nSegmentation Type: {seg_type}")
        
        # Open the GIF file
        self.gif = Image.open(gif_path)
        self.frames = [frame.copy() for frame in ImageSequence.Iterator(self.gif)]
        self.n_frames = len(self.frames)
        self.frame_index = 0
        
        # Display first frame
        self.update_frame()
    
    def update_frame(self):
        """Display the current frame"""
        if not hasattr(self, 'frames') or not self.frames:
            return
            
        # Get the current frame and convert to PhotoImage
        frame = self.frames[self.frame_index]
        
        # Resize if needed
        window_width = self.image_frame.winfo_width()
        window_height = self.image_frame.winfo_height()
        
        if window_width > 1 and window_height > 1:  # Ensure window is properly sized
            # Calculate new dimensions while maintaining aspect ratio
            img_width, img_height = frame.size
            ratio = min(window_width/img_width, window_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize the image
            frame = frame.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter-compatible image
        self.tk_image = ImageTk.PhotoImage(frame)
        self.image_label.config(image=self.tk_image)
        
        # Update title with frame info
        self.root.title(f"Attention GIF Viewer - Frame {self.frame_index+1}/{self.n_frames}")
        
        # Schedule next frame if playing
        if self.playing:
            effective_delay = int(self.delay / self.speed_var.get())
            self.root.after(effective_delay, self.next_frame)
    
    def next_frame(self):
        """Show the next frame"""
        if not hasattr(self, 'frames') or not self.frames:
            return
            
        self.frame_index = (self.frame_index + 1) % self.n_frames
        self.update_frame()
    
    def prev_frame(self):
        """Show the previous frame"""
        if not hasattr(self, 'frames') or not self.frames:
            return
            
        self.frame_index = (self.frame_index - 1) % self.n_frames
        self.update_frame()
    
    def next_file(self):
        """Load the next GIF file"""
        self.current_index = (self.current_index + 1) % len(self.gif_files)
        self.load_current_gif()
    
    def prev_file(self):
        """Load the previous GIF file"""
        self.current_index = (self.current_index - 1) % len(self.gif_files)
        self.load_current_gif()
    
    def toggle_play(self):
        """Toggle playback state"""
        self.playing = not self.playing
        if self.playing:
            self.next_frame()  # Start playing
    
    def update_speed(self, *args):
        """Update the playback speed"""
        # The speed control only affects future frame scheduling
        pass

def main():
    parser = argparse.ArgumentParser(description="View attention map GIFs")
    parser.add_argument("--dir", type=str, help="Directory containing GIF files", 
                        default=None)
    args = parser.parse_args()
    
    # If no directory specified, use the most recent result directory
    if args.dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        if os.path.exists(results_dir):
            # Find directories starting with 'cam_gif_visualizations_'
            cam_dirs = [d for d in os.listdir(results_dir) 
                       if os.path.isdir(os.path.join(results_dir, d)) 
                       and d.startswith('cam_gif_visualizations_')]
            
            if cam_dirs:
                # Sort by name (which includes timestamp)
                cam_dirs.sort(reverse=True)
                args.dir = os.path.join(results_dir, cam_dirs[0])
                print(f"Using most recent results directory: {args.dir}")
            else:
                print("No visualization directories found in results folder")
                return
        else:
            print(f"Results directory not found: {results_dir}")
            return
    
    # Create and run the viewer
    root = tk.Tk()
    root.geometry("1024x768")
    viewer = GifViewer(root, args.dir)
    root.mainloop()

if __name__ == "__main__":
    main() 