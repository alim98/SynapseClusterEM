import os
import numpy as np
import pandas as pd
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

from newdl.dataset2 import SynapseDataset
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
from synapse.utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sample_fig_compare_crop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sample_fig_compare_crop")

# Predefined samples for visualization
fixed_samples = [
    {"id": 1, "bbox_name": "bbox1", "Var1": "non_spine_synapse_004", "slice_number": 25},
    {"id": 2, "bbox_name": "bbox1", "Var1": "non_spine_synapse_006", "slice_number": 40},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-08-29_Vera_Broens_085", "slice_number": 33},
    {"id": 4, "bbox_name": "bbox2", "Var1": "explorative_2024-08-28_Cora_Wolter_031", "slice_number": 43},
    {"id": 5, "bbox_name": "bbox3", "Var1": "non_spine_synapse_036", "slice_number": 41},
    {"id": 6, "bbox_name": "bbox3", "Var1": "non_spine_synapse_018", "slice_number": 41},
    {"id": 7, "bbox_name": "bbox4", "Var1": "explorative_2024-08-03_Ali_Karimi_022_5_238", "slice_number": 56},
    {"id": 8, "bbox_name": "bbox5", "Var1": "non_spine_synapse_033", "slice_number": 48},
    {"id": 9, "bbox_name": "bbox5", "Var1": "non_spine_synapse_045", "slice_number": 40},
    {"id": 10, "bbox_name": "bbox6", "Var1": "spine_synapse_070", "slice_number": 37},
    {"id": 11, "bbox_name": "bbox6", "Var1": "spine_synapse_021", "slice_number": 30},
    {"id": 12, "bbox_name": "bbox7", "Var1": "non_spine_synapse_013", "slice_number": 25},
]

def create_combined_frames(frames_standard, frames_intelligent, title_standard="Standard Cropping", title_intelligent="Intelligent Cropping"):
    """
    Combine frames from standard and intelligent cropping side by side with labels
    with improved visual styling and larger size
    """
    combined_frames = []
    for i in range(len(frames_standard)):
        # Create blank images with the correct size
        standard_frame = Image.fromarray(frames_standard[i])
        intelligent_frame = Image.fromarray(frames_intelligent[i])
        
        # Get dimensions
        width, height = standard_frame.size
        
        # Add margins and header space
        margin = 20  # Increased margin between images
        header_height = 50  # Increased header height for better title display
        total_width = (width * 2) + margin  # Add margin between images
        total_height = height + header_height + 10  # Added a bit of bottom padding
        
        # Create a new image with proper dimensions and gray background
        combined = Image.new('RGB', (total_width, total_height), color=(240, 240, 240))
        
        # Add a border around each image
        border_color = (200, 200, 200)
        for x in range(width):
            for y in range(height):
                if x < 2 or x >= width-2 or y < 2 or y >= height-2:
                    # Draw border pixels for standard frame
                    combined.putpixel((x, y+header_height), border_color)
                    # Draw border pixels for intelligent frame
                    combined.putpixel((x+width+margin, y+header_height), border_color)
        
        # Paste images with appropriate offsets
        combined.paste(standard_frame, (0, header_height))
        combined.paste(intelligent_frame, (width + margin, header_height))
        
        # Create a header background
        header_bg = Image.new('RGB', (total_width, header_height), color=(230, 230, 230))
        combined.paste(header_bg, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(combined)
        
        # Try to use a suitable font with smaller size
        try:
            title_font = ImageFont.truetype("arial.ttf", 9)
            frame_font = ImageFont.truetype("arial.ttf", 7)
        except IOError:
            try:
                title_font = ImageFont.truetype("DejaVuSans.ttf", 9)
                frame_font = ImageFont.truetype("DejaVuSans.ttf", 7)
            except IOError:
                title_font = ImageFont.load_default()
                frame_font = title_font
        
        # Draw centered titles with improved visibility
        def draw_centered_text(text, x_center, y_center, max_width, font=title_font):
            # Get text size using getbbox (newer Pillow versions) or fallback method
            try:
                # For newer Pillow versions
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except (AttributeError, TypeError):
                try:
                    # For older Pillow versions
                    text_width, text_height = draw.textsize(text, font=font)
                except (AttributeError, TypeError):
                    # Last resort fallback
                    text_width, text_height = font.getsize(text)
            
            # Ensure text fits by truncating if necessary
            if text_width > max_width:
                while text_width > max_width and len(text) > 3:
                    text = text[:-1]  # Remove last character
                    try:
                        # Try with getbbox first
                        bbox = font.getbbox(text + "...")
                        text_width = bbox[2] - bbox[0]
                    except (AttributeError, TypeError):
                        try:
                            # Fallback to textsize
                            text_width, _ = draw.textsize(text + "...", font=font)
                        except (AttributeError, TypeError):
                            # Last resort
                            text_width, _ = font.getsize(text + "...")
                text = text + "..."
            
            # Calculate position
            x = x_center - (text_width // 2)
            y = y_center - (text_height // 2)
            
            # Draw text with a subtle shadow for better visibility
            shadow_color = (180, 180, 180)
            draw.text((x+2, y+2), text, fill=shadow_color, font=font)
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
        
        # Draw centered text for each title
        draw_centered_text(title_standard, width // 2, header_height // 2, width - 20)
        draw_centered_text(title_intelligent, width + margin + (width // 2), header_height // 2, width - 20)
        
        # Add a more visible separator line
        draw.line([(width + margin//2, 0), (width + margin//2, total_height)], fill=(180, 180, 180), width=2)
        
        # Add frame index
        frame_number_text = f"Frame {i+1}/{len(frames_standard)}"
        try:
            # Try with getbbox first
            bbox = frame_font.getbbox(frame_number_text)
            text_width = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            try:
                # Fallback to textsize
                text_width, _ = draw.textsize(frame_number_text, font=frame_font)
            except (AttributeError, TypeError):
                # Last resort
                text_width, _ = frame_font.getsize(frame_number_text)
            
        draw.text((total_width - text_width - 10, 10), frame_number_text, fill=(80, 80, 80), font=frame_font)
        
        combined_frames.append(np.array(combined))
    
    return combined_frames

def visualize_comparison(syn_df, bbox_name, var1, vol_data_dict, save_gifs_dir, segmentation_type, 
                         processor, subvol_size=80, num_frames=80, alpha=0.3, presynapse_weight=0.5,
                         normalize_presynapse_size=False, target_percentage=None, size_tolerance=0.1):
    """
    Create and save a side-by-side GIF comparison of standard and intelligent cropping
    
    Args:
        syn_df (pd.DataFrame): DataFrame containing synapse information
        bbox_name (str): Name of the bounding box
        var1 (str): Var1 value to identify the specific synapse
        vol_data_dict (dict): Dictionary of volume data
        save_gifs_dir (str): Directory to save GIFs
        segmentation_type (int): Type of segmentation to use
        processor: Processor for the images
        subvol_size (int): Size of the subvolume cube
        num_frames (int): Number of frames to include in the GIF
        alpha (float): Alpha value for transparent overlays
        presynapse_weight (float): Weight to shift center toward presynapse (0.0-1.0)
        normalize_presynapse_size (bool): Whether to normalize the presynapse size
        target_percentage (float, optional): Target percentage for presynapse size (None = use mean)
        size_tolerance (float): Tolerance range for presynapse size normalization
    """
    # Get volume data
    raw_vol, seg_vol, add_mask_vol = vol_data_dict.get(bbox_name, (None, None, None))
    if raw_vol is None:
        logger.error(f"No volume data found for {bbox_name}")
        return
    
    # Get synapse data
    synapse_data = syn_df[(syn_df['bbox_name'] == bbox_name) & (syn_df['Var1'] == var1)]
    if len(synapse_data) == 0:
        logger.error(f"No synapse data found for {bbox_name} - {var1}")
        return
    
    # Get the first matching synapse data
    syn_info = synapse_data.iloc[0]
    
    # Extract coordinates
    central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
    
    # Initialize data loader
    data_loader = SynapseDataLoader("", "", "")
    
    # Create standard (non-intelligent) cropped cube
    standard_cube = data_loader.create_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=segmentation_type,
        subvolume_size=subvol_size,
        alpha=alpha,
        bbox_name=bbox_name,
        normalize_across_volume=True,
        smart_crop=False,
        presynapse_weight=presynapse_weight,
        normalize_presynapse_size=normalize_presynapse_size,
        target_percentage=target_percentage,
        size_tolerance=size_tolerance
    )
    
    # Create intelligently cropped cube
    intelligent_cube = data_loader.create_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=segmentation_type,
        subvolume_size=subvol_size,
        alpha=alpha,
        bbox_name=bbox_name,
        normalize_across_volume=True,
        smart_crop=True,
        presynapse_weight=presynapse_weight,
        normalize_presynapse_size=normalize_presynapse_size,
        target_percentage=target_percentage,
        size_tolerance=size_tolerance
    )
    
    # Capture shift information from logs
    log_lines = [line for line in open("sample_fig_compare_crop.log").readlines() if "Smart cropping" in line and var1 in line and f"weight {presynapse_weight}" in line]
    shift_info = ""
    for line in log_lines:
        if "Shifted by:" in line:
            shift_info = line.split("Shifted by:")[1].strip()
            break
    
    # Extract frames from the standard cube
    standard_frames = [standard_cube[..., z] for z in range(standard_cube.shape[3])]
    
    # Extract frames from the intelligent cube
    intelligent_frames = [intelligent_cube[..., z] for z in range(intelligent_cube.shape[3])]
    
    # Ensure we have the correct number of frames
    if len(standard_frames) < num_frames:
        standard_frames += [standard_frames[-1]] * (num_frames - len(standard_frames))
    elif len(standard_frames) > num_frames:
        indices = np.linspace(0, len(standard_frames)-1, num_frames, dtype=int)
        standard_frames = [standard_frames[i] for i in indices]
    
    if len(intelligent_frames) < num_frames:
        intelligent_frames += [intelligent_frames[-1]] * (num_frames - len(intelligent_frames))
    elif len(intelligent_frames) > num_frames:
        indices = np.linspace(0, len(intelligent_frames)-1, num_frames, dtype=int)
        intelligent_frames = [intelligent_frames[i] for i in indices]
    
    # Convert to uint8 for GIF saving without per-frame normalization
    # First convert lists to numpy arrays for consistent processing
    standard_frames_array = np.stack(standard_frames)
    intelligent_frames_array = np.stack(intelligent_frames)
    
    # Find global min and max across all frames to ensure consistent grayscale
    standard_min = standard_frames_array.min()
    standard_max = standard_frames_array.max()
    intelligent_min = intelligent_frames_array.min()
    intelligent_max = intelligent_frames_array.max()
    
    logger.info(f"Standard cube range: min={standard_min:.4f}, max={standard_max:.4f}")
    logger.info(f"Intelligent cube range: min={intelligent_min:.4f}, max={intelligent_max:.4f}")
    
    # Create output frames using consistent scaling
    enhanced_standard_frames = []
    for frame in standard_frames:
        if standard_max > standard_min:  # Avoid division by zero
            # Use global normalization across all frames
            scaled = (frame - standard_min) / (standard_max - standard_min)
            enhanced_standard_frames.append((scaled * 255).astype(np.uint8))
        else:
            enhanced_standard_frames.append(np.zeros_like(frame, dtype=np.uint8))
    
    enhanced_intelligent_frames = []
    for frame in intelligent_frames:
        if intelligent_max > intelligent_min:  # Avoid division by zero
            # Use global normalization across all frames
            scaled = (frame - intelligent_min) / (intelligent_max - intelligent_min)
            enhanced_intelligent_frames.append((scaled * 255).astype(np.uint8))
        else:
            enhanced_intelligent_frames.append(np.zeros_like(frame, dtype=np.uint8))
    
    # Create separate GIFs
    size_normalized_suffix = "_normalized" if normalize_presynapse_size else ""
    standard_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_standard{size_normalized_suffix}.gif")
    intelligent_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_intelligent_w{presynapse_weight}{size_normalized_suffix}.gif")
    
    # Save individual GIFs
    try:
        logger.info(f"Saving standard GIF to {standard_gif_path}")
        imageio.mimsave(standard_gif_path, enhanced_standard_frames, fps=8)  # Slower frame rate
        
        logger.info(f"Saving intelligent GIF to {intelligent_gif_path}")
        imageio.mimsave(intelligent_gif_path, enhanced_intelligent_frames, fps=8)  # Slower frame rate
    except Exception as e:
        logger.error(f"Failed to save individual GIFs: {e}")
    
    # Create descriptive titles
    title_standard = "Standard Cropping"
    if normalize_presynapse_size:
        title_standard += " (Size Norm)"
    
    title_intelligent = f"Intelligent Cropping (w={presynapse_weight})"
    if normalize_presynapse_size:
        title_intelligent += ", Size Norm"
    if shift_info:
        title_intelligent += f" - Shift: {shift_info}"
    
    # Create and save side-by-side comparison GIF
    combined_frames = create_combined_frames(
        enhanced_standard_frames, 
        enhanced_intelligent_frames,
        title_standard, 
        title_intelligent
    )
    
    combined_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_comparison_w{presynapse_weight}{size_normalized_suffix}.gif")
    
    try:
        logger.info(f"Saving combined comparison GIF to {combined_gif_path}")
        imageio.mimsave(combined_gif_path, combined_frames, fps=8)  # Slower frame rate
        logger.info("GIF saved successfully")
    except Exception as e:
        logger.error(f"Failed to save combined GIF: {e}")
        
    # Create an info file with details of the comparison
    info_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_info{size_normalized_suffix}.txt")
    with open(info_path, 'w') as f:
        f.write(f"Sample: {var1} from {bbox_name}\n")
        f.write(f"Segmentation Type: {segmentation_type}\n")
        f.write(f"Presynapse Weight: {presynapse_weight}\n")
        f.write(f"Size Normalization: {'Enabled' if normalize_presynapse_size else 'Disabled'}\n")
        if normalize_presynapse_size:
            f.write(f"Target Percentage: {target_percentage if target_percentage is not None else 'Auto (Mean)'}\n")
            f.write(f"Size Tolerance: ±{size_tolerance*100:.1f}%\n")
        f.write(f"Original Center: {central_coord}\n")
        if shift_info:
            f.write(f"Shift Vector: {shift_info}\n")
        f.write(f"Standard Range: min={standard_min:.4f}, max={standard_max:.4f}\n")
        f.write(f"Intelligent Range: min={intelligent_min:.4f}, max={intelligent_max:.4f}\n")

def create_all_combinations_comparison(sample_key, frame_paths, save_gifs_dir, weights):
    """
    Create a comprehensive comparison showing all combinations of:
    1. Original/standard cropping
    2. Intelligent cropping with different weights
    3. Both with and without size normalization
    
    This creates a 2x4 grid visualization:
    - Top row: Without size normalization
      [Standard | Intelligent w=0.3 | Intelligent w=0.5 | Intelligent w=0.7]
    - Bottom row: With size normalization
      [Standard | Intelligent w=0.3 | Intelligent w=0.5 | Intelligent w=0.7]
    
    Args:
        sample_key: Identifier string for the sample
        frame_paths: Dictionary mapping all types of frames to their paths
        save_gifs_dir: Directory to save the output
        weights: List of weights used for intelligent cropping
    
    Returns:
        Path to the saved GIF file
    """
    logger.info(f"Creating all combinations comparison for {sample_key}")
    
    # Check if we have all the required frames
    missing_frames = []
    
    # Check standard frames
    for key in ['standard', 'standard_normalized']:
        if key not in frame_paths or not frame_paths[key]:
            missing_frames.append(key)
    
    # Check intelligent frames with all weights
    for weight in weights:
        for norm in [False, True]:
            norm_suffix = "_normalized" if norm else ""
            key = f"intelligent_w{weight}{norm_suffix}"
            if key not in frame_paths or not frame_paths[key]:
                missing_frames.append(key)
    
    if missing_frames:
        logger.error(f"Missing required frames for all combinations comparison: {missing_frames}")
        return None
    
    # Load all the GIFs
    gif_frames = {}
    num_frames = None
    
    # First determine dimensions from the standard crop
    standard_gif_path = frame_paths['standard']
    logger.info(f"Loading standard GIF from {standard_gif_path}")
    standard_gif = Image.open(standard_gif_path)
    frame_width = standard_gif.width
    frame_height = standard_gif.height
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Load frames for all types
    for key, gif_path in frame_paths.items():
        logger.info(f"Loading {key} from {gif_path}")
        gif = Image.open(gif_path)
        frames = []
        try:
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            gif_frames[key] = frames
            logger.info(f"Loaded {len(frames)} frames for {key}")
            # Use the first valid number of frames
            if num_frames is None:
                num_frames = len(frames)
        except Exception as e:
            logger.error(f"Error loading frames from {gif_path}: {e}")
    
    if num_frames is None:
        logger.error("Couldn't determine number of frames")
        return None
    
    # Calculate composition dimensions with enhanced spacing and layout
    margin = 15  # Margin between cells
    header_height = 40  # Height for headers
    title_height = 60  # Height for main title
    footer_height = 40  # Height for bottom info
    
    # Format: 2 rows (unnormalized/normalized) x (1 + len(weights)) columns (standard + weights)
    num_columns = len(weights) + 1  # standard + all weights
    
    # Calculate total width and height with margins
    composite_width = (frame_width * num_columns) + (margin * (num_columns + 1))
    composite_height = (frame_height * 2) + (margin * 3) + header_height * 2 + title_height + footer_height
    
    # Define colors for weight indicators and backgrounds
    weight_colors = {
        0.3: (220, 50, 50),    # Red for low weight
        0.5: (50, 50, 220),    # Blue for medium weight
        0.7: (50, 180, 50)     # Green for high weight
    }
    
    # Background colors
    bg_color = (240, 240, 240)  # Light gray background
    header_color = (30, 30, 30)  # Dark header
    norm_header_color = (60, 40, 80)  # Purple-ish for normalization header
    weight_bg_colors = {
        0.3: (255, 220, 220),  # Light red background
        0.5: (220, 220, 255),  # Light blue background
        0.7: (220, 255, 220)   # Light green background
    }
    
    # Font setup with fallbacks
    try:
        title_font = ImageFont.truetype("arial.ttf", 12)
        header_font = ImageFont.truetype("arial.ttf", 9) 
        label_font = ImageFont.truetype("arial.ttf", 8)
        small_font = ImageFont.truetype("arial.ttf", 7)
        logger.info("Using Arial font for text")
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 7)
            logger.info("Using DejaVuSans font for text")
        except:
            title_font = ImageFont.load_default()
            header_font = title_font
            label_font = title_font
            small_font = title_font
            logger.info("Using default font for text")
    
    # Create colored border function with improved visibility
    def add_colored_border(img, color, width=4):
        """Add a colored border to an image with a smooth gradient effect."""
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        # Draw border rectangles with decreasing width for gradient effect
        opacity_step = 255 // width
        for i in range(width):
            opacity = 255 - (i * opacity_step)
            border_color = (*color, opacity)
            draw.rectangle(
                [(i, i), (img_width - i - 1, img_height - i - 1)],
                outline=border_color,
                width=2 if i == 0 else 1
            )
        return img
    
    # Create composite frames
    composite_frames = []
    for i in range(num_frames):
        # Create a new image with proper background
        composite = Image.new('RGBA', (composite_width, composite_height), color=bg_color)
        draw = ImageDraw.Draw(composite)
        
        # Add main title with sample info
        title_bg_rect = (0, 0, composite_width, title_height)
        draw.rectangle(title_bg_rect, fill=header_color)
        title_text = f"All Combinations Comparison - {sample_key.replace('_', ' ').title()}"
        draw_centered_text(draw, title_text, composite_width // 2, title_height // 2, title_font)
        
        # Draw arrows showing weight progression
        arrow_y = title_height + 25
        arrow_start_x = margin + frame_width + margin
        arrow_end_x = composite_width - margin - frame_width // 2
        composite = add_weight_arrow(composite, arrow_y, arrow_start_x, arrow_end_x, (100, 100, 100))
        draw = ImageDraw.Draw(composite)  # Redraw after modification
        draw_centered_text(draw, "Increasing Presynapse Weight →", (arrow_start_x + arrow_end_x) // 2, 
                          arrow_y - 20, small_font, fill=(50, 50, 50))
        
        # Calculate starting positions for frames
        unnorm_start_y = title_height + header_height + margin
        norm_start_y = unnorm_start_y + frame_height + margin + header_height
        
        # Add section headers for unnormalized row
        unnorm_header_rect = (0, title_height, composite_width, title_height + header_height)
        draw.rectangle(unnorm_header_rect, fill=(50, 50, 50))
        draw_centered_text(draw, "Standard Cropping + Intelligent Cropping (Without Size Normalization)", 
                          composite_width // 2, title_height + header_height // 2, header_font)
        
        # Add section headers for normalized row
        norm_header_rect = (0, unnorm_start_y + frame_height + margin, 
                           composite_width, unnorm_start_y + frame_height + margin + header_height)
        draw.rectangle(norm_header_rect, fill=norm_header_color)
        draw_centered_text(draw, "Standard Cropping + Intelligent Cropping (With Size Normalization - Target: 15%)", 
                          composite_width // 2, unnorm_start_y + frame_height + margin + header_height // 2, 
                          header_font)
        
        # Add standard crop - top row (unnormalized)
        if 'standard' in gif_frames and i < len(gif_frames['standard']):
            # Get the frame
            std_frame = gif_frames['standard'][i].copy()
            
            # Add border
            std_frame = add_colored_border(std_frame, (0, 0, 0), width=3)
            
            # Calculate position
            x_pos = margin
            y_pos = unnorm_start_y
            
            # Add background for cell
            cell_bg_rect = (x_pos - 5, y_pos - 5, 
                           x_pos + frame_width + 5, y_pos + frame_height + 5)
            draw.rectangle(cell_bg_rect, fill=(230, 230, 230))
            
            # Paste the frame
            composite.paste(std_frame, (x_pos, y_pos))
            
            # Add label
            label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                            x_pos + frame_width, y_pos + frame_height)
            draw.rectangle(label_bg_rect, fill=(0, 0, 0, 180))
            draw_centered_text(draw, "Standard Crop", x_pos + frame_width // 2, 
                              y_pos + frame_height - 15, label_font)
        
        # Add standard crop - bottom row (normalized)
        if 'standard_normalized' in gif_frames and i < len(gif_frames['standard_normalized']):
            # Get the frame
            std_norm_frame = gif_frames['standard_normalized'][i].copy()
            
            # Add border
            std_norm_frame = add_colored_border(std_norm_frame, (0, 0, 0), width=3)
            
            # Calculate position
            x_pos = margin
            y_pos = norm_start_y
            
            # Add background for cell
            cell_bg_rect = (x_pos - 5, y_pos - 5, 
                           x_pos + frame_width + 5, y_pos + frame_height + 5)
            draw.rectangle(cell_bg_rect, fill=(230, 230, 230))
            
            # Paste the frame
            composite.paste(std_norm_frame, (x_pos, y_pos))
            
            # Add label
            label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                            x_pos + frame_width, y_pos + frame_height)
            draw.rectangle(label_bg_rect, fill=(0, 0, 0, 180))
            draw_centered_text(draw, "Standard + Size Norm", x_pos + frame_width // 2, 
                              y_pos + frame_height - 15, label_font)
        
        # Add intelligent crops with different weights - both rows
        for idx, weight in enumerate(weights):
            # Calculate positions
            x_pos = margin + (idx + 1) * (frame_width + margin)
            
            # Colors for this weight
            weight_color = weight_colors[weight]
            weight_bg = weight_bg_colors[weight]
            
            # Add background coloring for column
            col_bg_rect = (x_pos - 5, title_height + header_height, 
                          x_pos + frame_width + 5, composite_height - footer_height)
            draw.rectangle(col_bg_rect, fill=(weight_bg[0], weight_bg[1], weight_bg[2], 50))
            
            # Process unnormalized intelligent crop
            key = f"intelligent_w{weight}"
            if key in gif_frames and i < len(gif_frames[key]):
                # Get the frame and add colored border
                intel_frame = gif_frames[key][i].copy()
                intel_frame = add_colored_border(intel_frame, weight_color, width=5)
                
                # Add background for cell
                cell_bg_rect = (x_pos - 5, unnorm_start_y - 5, 
                               x_pos + frame_width + 5, unnorm_start_y + frame_height + 5)
                draw.rectangle(cell_bg_rect, fill=weight_bg)
                
                # Paste frame
                y_pos = unnorm_start_y
                composite.paste(intel_frame, (x_pos, y_pos))
                
                # Add label with weight color
                label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                                x_pos + frame_width, y_pos + frame_height)
                draw.rectangle(label_bg_rect, fill=(weight_color[0], weight_color[1], weight_color[2], 180))
                draw_centered_text(draw, f"Intelligent (w={weight})", x_pos + frame_width // 2, 
                                  y_pos + frame_height - 15, label_font, fill=(255, 255, 255))
            
            # Process normalized intelligent crop
            key = f"intelligent_w{weight}_normalized"
            if key in gif_frames and i < len(gif_frames[key]):
                # Get the frame and add colored border
                intel_norm_frame = gif_frames[key][i].copy()
                intel_norm_frame = add_colored_border(intel_norm_frame, weight_color, width=5)
                
                # Add background for cell
                cell_bg_rect = (x_pos - 5, norm_start_y - 5, 
                               x_pos + frame_width + 5, norm_start_y + frame_height + 5)
                draw.rectangle(cell_bg_rect, fill=weight_bg)
                
                # Paste frame
                y_pos = norm_start_y
                composite.paste(intel_norm_frame, (x_pos, y_pos))
                
                # Add label with weight color
                label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                                x_pos + frame_width, y_pos + frame_height)
                draw.rectangle(label_bg_rect, fill=(weight_color[0], weight_color[1], weight_color[2], 180))
                draw_centered_text(draw, f"Intelligent + Size Norm", x_pos + frame_width // 2, 
                                  y_pos + frame_height - 15, label_font, fill=(255, 255, 255))
        
        # Add frame counter and metadata at bottom
        footer_rect = (0, composite_height - footer_height, composite_width, composite_height)
        draw.rectangle(footer_rect, fill=(50, 50, 50))
        
        # Add frame counter
        frame_text = f"Frame {i+1}/{num_frames}"
        draw_centered_text(draw, frame_text, 100, composite_height - footer_height//2, small_font)
        
        # Add legend
        legend_text = "Red = Weight 0.3 | Blue = Weight 0.5 | Green = Weight 0.7"
        draw_centered_text(draw, legend_text, composite_width - 250, 
                          composite_height - footer_height//2, small_font)
        
        composite_frames.append(composite)
    
    # Save the composite GIF
    output_path = os.path.join(save_gifs_dir, f"{sample_key}_all_combinations.gif")
    logger.info(f"Saving all combinations comparison to {output_path} with {len(composite_frames)} frames")
    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        optimize=False,
        duration=200,
        loop=0
    )
    return output_path

# Function to add an arrow indicating the effect of weight
def add_weight_arrow(img, y_pos, x_start, x_end, color):
    """Add an arrow indicating the direction of weight effect."""
    draw = ImageDraw.Draw(img)
    
    # Draw the line
    draw.line([(x_start, y_pos), (x_end, y_pos)], fill=color, width=3)
    
    # Draw arrowhead
    arrow_size = 10
    draw.polygon([(x_end, y_pos), (x_end - arrow_size, y_pos - arrow_size//2), 
                  (x_end - arrow_size, y_pos + arrow_size//2)], fill=color)
    
    return img

def draw_centered_text(draw, text, x, y, font, fill=(255, 255, 255)):
    """Draw text centered on a position using PIL."""
    # Use font.getbbox() instead of draw.textsize() (which is deprecated)
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Fallback for older PIL versions
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except:
            # Another fallback - approximate
            text_width = len(text) * font.size * 0.6
            text_height = font.size
    
    draw.text(
        (x - text_width // 2, y - text_height // 2),
        text,
        font=font,
        fill=fill
    )

def create_multi_weight_comparison(sample_key, weight_frames, save_gifs_dir, weights):
    """
    Create a side-by-side comparison of different presynapse weights.
    
    Args:
        sample_key: Identifier string for the sample
        weight_frames: Dictionary mapping weights to GIF paths
        save_gifs_dir: Directory to save the output
        weights: List of weights used
    """
    logger.info(f"Creating multi-weight comparison for {sample_key}")
    logger.info(f"Available weight frames: {weight_frames.keys()}")
    logger.info(f"Expected weights: {weights}")
    
    # Check if we have frames for all weights
    if not all(w in weight_frames for w in weights) or 'standard' not in weight_frames:
        logger.error(f"Missing required frames for multi-weight comparison")
        missing = [w for w in weights if w not in weight_frames]
        if 'standard' not in weight_frames:
            missing.append('standard')
        logger.error(f"Missing weights: {missing}")
        return
    
    # Load all the GIFs
    gif_frames = {}
    num_frames = None
    
    # First determine dimensions from the standard crop
    standard_gif_path = weight_frames['standard']
    logger.info(f"Loading standard GIF from {standard_gif_path}")
    try:
        standard_gif = Image.open(standard_gif_path)
        frame_width = standard_gif.width
        frame_height = standard_gif.height
        logger.info(f"Standard GIF dimensions: {frame_width}x{frame_height}")
    except Exception as e:
        logger.error(f"Error opening standard GIF: {e}")
        return
    
    # Load frames for standard and each weight
    for key, gif_path in weight_frames.items():
        if key == 'standard' or key in weights:
            logger.info(f"Loading frames for {key} from {gif_path}")
            try:
                gif = Image.open(gif_path)
                frames = []
                for frame in ImageSequence.Iterator(gif):
                    frames.append(frame.copy())
                gif_frames[key] = frames
                logger.info(f"Loaded {len(frames)} frames for {key}")
                # Use the first valid number of frames
                if num_frames is None:
                    num_frames = len(frames)
            except Exception as e:
                logger.error(f"Error loading frames from {gif_path}: {e}")
    
    if num_frames is None:
        logger.error("Couldn't determine number of frames")
        return
    
    # Calculate composition dimensions
    # We'll have the standard crop and then one for each weight
    num_columns = len(weights) + 1  # standard + all weights
    composite_width = frame_width * num_columns
    composite_height = frame_height
    logger.info(f"Creating composite with dimensions: {composite_width}x{composite_height}")
    
    # Font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        logger.info("Using arial.ttf for text")
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            logger.info("Using DejaVuSans.ttf for text")
        except:
            font = ImageFont.load_default()
            logger.info("Using default font for text")
    
    # Create composite frames
    composite_frames = []
    for i in range(num_frames):
        logger.debug(f"Creating composite frame {i+1}/{num_frames}")
        composite = Image.new('RGB', (composite_width, composite_height))
        
        # Add standard crop
        if 'standard' in gif_frames and i < len(gif_frames['standard']):
            composite.paste(gif_frames['standard'][i], (0, 0))
            # Add label
            draw = ImageDraw.Draw(composite)
            draw_centered_text(draw, "Standard", frame_width // 2, 30, font)
        
        # Add intelligent crops with different weights
        for idx, weight in enumerate(weights):
            x_offset = (idx + 1) * frame_width
            if weight in gif_frames and i < len(gif_frames[weight]):
                composite.paste(gif_frames[weight][i], (x_offset, 0))
                # Add label
                draw = ImageDraw.Draw(composite)
                draw_centered_text(draw, f"Weight: {weight}", x_offset + frame_width // 2, 30, font)
        
        composite_frames.append(composite)
    
    # Save the composite GIF
    output_path = os.path.join(save_gifs_dir, f"{sample_key}_weight_comparison.gif")
    logger.info(f"Saving multi-weight comparison to {output_path}")
    try:
        composite_frames[0].save(
            output_path,
            save_all=True,
            append_images=composite_frames[1:],
            optimize=False,
            duration=200,
            loop=0
        )
        logger.info(f"Saved multi-weight comparison to {output_path}")
    except Exception as e:
        logger.error(f"Error saving multi-weight comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())

def create_size_normalization_comparison(sample_key, frame_paths, save_gifs_dir, weight):
    """
    Create a side-by-side comparison of crops with and without size normalization.
    
    Args:
        sample_key: Identifier string for the sample
        frame_paths: Dictionary mapping crop types to GIF paths
        save_gifs_dir: Directory to save the output
        weight: Presynapse weight used for intelligent crop
    """
    logger.info(f"Creating size normalization comparison for {sample_key}")
    
    # Check if we have all required frames
    required_types = ['standard', 'standard_normalized', 'intelligent', 'intelligent_normalized']
    if not all(t in frame_paths and frame_paths[t] for t in required_types):
        logger.error(f"Missing required frames for size normalization comparison")
        missing = [t for t in required_types if t not in frame_paths or not frame_paths[t]]
        logger.error(f"Missing types: {missing}")
        return
    
    # Load all the GIFs
    gif_frames = {}
    num_frames = None
    
    # First determine dimensions
    standard_gif = Image.open(frame_paths['standard'])
    frame_width = standard_gif.width
    frame_height = standard_gif.height
    
    # Load frames for all types
    for key, gif_path in frame_paths.items():
        gif = Image.open(gif_path)
        frames = []
        try:
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            gif_frames[key] = frames
            # Use the first valid number of frames
            if num_frames is None:
                num_frames = len(frames)
        except Exception as e:
            logger.error(f"Error loading frames from {gif_path}: {e}")
    
    if num_frames is None:
        logger.error("Couldn't determine number of frames")
        return
    
    # Calculate composition dimensions with enhanced spacing
    margin = 30  # Increased margin between cells
    header_height = 40  # Height for headers
    cell_padding = 10  # Padding inside each cell
    title_height = 50  # Main title height
    footer_height = 30  # Footer height
    
    # Calculate total dimensions with margins
    composite_width = (frame_width * 2) + (3 * margin)  # 2 columns with margins
    composite_height = (frame_height * 2) + (3 * margin) + title_height + footer_height + (2 * header_height)  # 2 rows with margins
    
    # Font setup with fallbacks - using smaller font sizes
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        header_font = ImageFont.truetype("arial.ttf", 12)
        label_font = ImageFont.truetype("arial.ttf", 10)
        small_font = ImageFont.truetype("arial.ttf", 8)
        logger.info("Using Arial font for text")
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            logger.info("Using DejaVuSans font for text")
        except:
            title_font = ImageFont.load_default()
            header_font = title_font
            label_font = title_font
            small_font = title_font
            logger.info("Using default font for text")
    
    # Create a function to add a colored border to an image
    def add_colored_border(img, color, width=2):
        """Add a colored border to an image."""
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        # Draw border rectangles
        for i in range(width):
            draw.rectangle(
                [(i, i), (img_width - i - 1, img_height - i - 1)],
                outline=color,
                width=1
            )
        return img
    
    # Create composite frames
    composite_frames = []
    for i in range(num_frames):
        # Create a new image with proper dimensions and gray background
        composite = Image.new('RGB', (composite_width, composite_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(composite)
        
        # Add main title with sample info
        title_bg_rect = (0, 0, composite_width, title_height)
        draw.rectangle(title_bg_rect, fill=(30, 30, 30))
        title_text = f"Size Normalization Comparison - {sample_key.replace('_', ' ').title()}"
        draw_centered_text(draw, title_text, composite_width // 2, title_height // 2, title_font)
        
        # Define row headers
        row_headers = [
            "Without Size Normalization",
            "With Size Normalization (Target: 15%)"
        ]
        
        # Define column headers
        col_headers = [
            "Standard Cropping",
            f"Intelligent Cropping (w={weight})"
        ]
        
        # Draw column headers
        for col in range(2):
            header_x = margin + (col * (frame_width + margin)) + frame_width // 2
            header_y = title_height + header_height // 2
            draw_centered_text(draw, col_headers[col], header_x, header_y, header_font)
        
        # Draw row headers (vertical text on the left)
        for row in range(2):
            header_x = margin // 2
            header_y = title_height + header_height + margin + (row * (frame_height + margin)) + frame_height // 2
            
            # Draw text rotated 90 degrees for row headers
            txt = Image.new('RGBA', (200, 30), (255, 255, 255, 0))
            d = ImageDraw.Draw(txt)
            d.text((0, 0), row_headers[row], font=header_font, fill=(50, 50, 50))
            
            # Rotate and paste the text
            rotated = txt.rotate(90, expand=True)
            composite.paste(rotated, (header_x - 15, header_y - 100), rotated)
        
        # Position in a 2x2 grid with proper margins:
        positions = {
            'standard': (margin, title_height + header_height + margin),
            'standard_normalized': (margin, title_height + header_height + (2 * margin) + frame_height),
            'intelligent': (margin * 2 + frame_width, title_height + header_height + margin),
            'intelligent_normalized': (margin * 2 + frame_width, title_height + header_height + (2 * margin) + frame_height)
        }
        
        labels = {
            'standard': "Standard",
            'standard_normalized': "Standard + Size Norm",
            'intelligent': f"Intelligent (w={weight})",
            'intelligent_normalized': f"Intelligent + Size Norm"
        }
        
        # Add all four visualizations to the composite
        for key, pos in positions.items():
            if key in gif_frames and i < len(gif_frames[key]):
                # Get the frame
                frame = gif_frames[key][i].copy()
                
                # Add border
                color = (0, 0, 0) if 'standard' in key else (50, 50, 220)  # Blue for intelligent
                frame = add_colored_border(frame, color)
                
                # Add cell background
                cell_bg_rect = (pos[0] - cell_padding, pos[1] - cell_padding,
                               pos[0] + frame_width + cell_padding, pos[1] + frame_height + cell_padding)
                bg_color = (230, 230, 230) if 'standard' in key else (220, 220, 255)  # Light blue for intelligent
                draw.rectangle(cell_bg_rect, fill=bg_color)
                
                # Paste the frame
                composite.paste(frame, pos)
                
                # Add label at the bottom of each cell
                label_bg_rect = (pos[0], pos[1] + frame_height - 20,
                                pos[0] + frame_width, pos[1] + frame_height)
                label_bg_color = (0, 0, 0, 180) if 'standard' in key else (50, 50, 220, 180)
                draw.rectangle(label_bg_rect, fill=label_bg_color)
                draw_centered_text(draw, labels[key], pos[0] + frame_width // 2,
                                 pos[1] + frame_height - 10, label_font, fill=(255, 255, 255))
        
        # Add frame counter at the bottom
        footer_rect = (0, composite_height - footer_height, composite_width, composite_height)
        draw.rectangle(footer_rect, fill=(50, 50, 50))
        frame_text = f"Frame {i+1}/{num_frames}"
        draw_centered_text(draw, frame_text, composite_width // 2, composite_height - footer_height // 2, small_font)
        
        composite_frames.append(composite)
    
    # Save the composite GIF
    output_path = os.path.join(save_gifs_dir, f"{sample_key}_normalization_comparison.gif")
    logger.info(f"Saving size normalization comparison to {output_path}")
    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        optimize=False,
        duration=200,
        loop=0
    )

    return output_path

def create_comprehensive_comparison(sample_key, frame_paths, save_gifs_dir, weights):
    """
    Create a comprehensive comparison showing all combinations of:
    1. Original/standard cropping
    2. Intelligent cropping with multiple weights
    3. Both normalized and unnormalized versions
    
    Args:
        sample_key: Identifier string for the sample
        frame_paths: Dictionary mapping all types of frames to their paths
        save_gifs_dir: Directory to save the output
        weights: List of weights used for intelligent cropping
    """
    logger.info(f"Creating comprehensive comparison for {sample_key}")
    
    # Check if we have all the required frames
    missing_frames = []
    
    # Check standard frames
    for key in ['standard', 'standard_normalized']:
        if key not in frame_paths or not frame_paths[key]:
            missing_frames.append(key)
    
    # Check intelligent frames with all weights
    for weight in weights:
        for norm in [False, True]:
            norm_suffix = "_normalized" if norm else ""
            key = f"intelligent_w{weight}{norm_suffix}"
            if key not in frame_paths or not frame_paths[key]:
                missing_frames.append(key)
    
    if missing_frames:
        logger.error(f"Missing required frames for comprehensive comparison: {missing_frames}")
        return
    
    # Load all the GIFs
    gif_frames = {}
    num_frames = None
    
    # First determine dimensions from the standard crop
    standard_gif_path = frame_paths['standard']
    logger.info(f"Loading standard GIF from {standard_gif_path}")
    standard_gif = Image.open(standard_gif_path)
    frame_width = standard_gif.width
    frame_height = standard_gif.height
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Load frames for all types
    for key, gif_path in frame_paths.items():
        logger.info(f"Loading {key} from {gif_path}")
        gif = Image.open(gif_path)
        frames = []
        try:
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            gif_frames[key] = frames
            logger.info(f"Loaded {len(frames)} frames for {key}")
            # Use the first valid number of frames
            if num_frames is None:
                num_frames = len(frames)
        except Exception as e:
            logger.error(f"Error loading frames from {gif_path}: {e}")
    
    if num_frames is None:
        logger.error("Couldn't determine number of frames")
        return
    
    # Calculate composition dimensions with enhanced spacing and layout
    margin = 15  # Margin between cells
    header_height = 40  # Height for headers
    title_height = 50  # Height for main title
    footer_height = 30  # Height for bottom info
    
    # Format: 2 rows (unnormalized/normalized) x (1 + len(weights)) columns (standard + weights)
    num_columns = len(weights) + 1  # standard + all weights
    
    # Calculate total width and height with margins
    composite_width = (frame_width * num_columns) + (margin * (num_columns + 1))
    composite_height = (frame_height * 2) + (margin * 3) + header_height * 2 + title_height + footer_height
    
    # Define colors for weight indicators (vibrant and distinct)
    weight_colors = {
        0.3: (220, 50, 50),    # Bright red for low weight
        0.5: (50, 50, 220),    # Bright blue for medium weight
        0.7: (50, 180, 50)     # Bright green for high weight
    }
    
    # Background colors
    bg_color = (240, 240, 240)  # Light gray background
    header_color = (30, 30, 30)  # Dark header
    weight_bg_colors = {
        0.3: (255, 220, 220),  # Light red background
        0.5: (220, 220, 255),  # Light blue background
        0.7: (220, 255, 220)   # Light green background
    }
    
    # Font setup with fallbacks
    try:
        title_font = ImageFont.truetype("arial.ttf", 12)
        header_font = ImageFont.truetype("arial.ttf", 9) 
        label_font = ImageFont.truetype("arial.ttf", 8)
        small_font = ImageFont.truetype("arial.ttf", 7)
        logger.info("Using Arial font for text")
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 7)
            logger.info("Using DejaVuSans font for text")
        except:
            title_font = ImageFont.load_default()
            header_font = title_font
            label_font = title_font
            small_font = title_font
            logger.info("Using default font for text")
    
    # Create colored border function with improved visibility
    def add_colored_border(img, color, width=4):
        """Add a colored border to an image with a smooth gradient effect."""
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        # Draw border rectangles with decreasing width for gradient effect
        opacity_step = 255 // width
        for i in range(width):
            opacity = 255 - (i * opacity_step)
            border_color = (*color, opacity)
            draw.rectangle(
                [(i, i), (img_width - i - 1, img_height - i - 1)],
                outline=border_color,
                width=2 if i == 0 else 1
            )
        return img
    
    # Create composite frames
    composite_frames = []
    for i in range(num_frames):
        # Create a new image with proper background
        composite = Image.new('RGBA', (composite_width, composite_height), color=bg_color)
        draw = ImageDraw.Draw(composite)
        
        # Add main title with sample info
        title_bg_rect = (0, 0, composite_width, title_height)
        draw.rectangle(title_bg_rect, fill=header_color)
        title_text = f"Comprehensive Comparison - {sample_key.replace('_', ' ').title()}"
        draw_centered_text(draw, title_text, composite_width // 2, title_height // 2, title_font)
        
        # Draw arrows showing weight progression
        arrow_y = title_height + 25
        arrow_start_x = margin + frame_width + margin
        arrow_end_x = composite_width - margin - frame_width // 2
        composite = add_weight_arrow(composite, arrow_y, arrow_start_x, arrow_end_x, (100, 100, 100))
        draw = ImageDraw.Draw(composite)  # Redraw after modification
        draw_centered_text(draw, "Increasing Presynapse Weight →", (arrow_start_x + arrow_end_x) // 2, 
                          arrow_y - 20, small_font, fill=(50, 50, 50))
        
        # Calculate starting positions for frames
        unnorm_start_y = title_height + header_height + margin
        norm_start_y = unnorm_start_y + frame_height + margin + header_height
        
        # Add section headers for unnormalized row
        unnorm_header_rect = (0, title_height, composite_width, title_height + header_height)
        draw.rectangle(unnorm_header_rect, fill=(50, 50, 50))
        draw_centered_text(draw, "Unnormalized Crops", composite_width // 2, 
                          title_height + header_height // 2, header_font)
        
        # Add section headers for normalized row
        norm_header_rect = (0, unnorm_start_y + frame_height + margin, 
                           composite_width, unnorm_start_y + frame_height + margin + header_height)
        draw.rectangle(norm_header_rect, fill=(50, 50, 50))
        draw_centered_text(draw, "Normalized Crops (Target Presynapse Size: 15%)", 
                          composite_width // 2, unnorm_start_y + frame_height + margin + header_height // 2, 
                          header_font)
        
        # Add standard crop - top row (unnormalized)
        if 'standard' in gif_frames and i < len(gif_frames['standard']):
            # Get the frame
            std_frame = gif_frames['standard'][i].copy()
            
            # Add border
            std_frame = add_colored_border(std_frame, (0, 0, 0), width=3)
            
            # Calculate position
            x_pos = margin
            y_pos = unnorm_start_y
            
            # Add background for cell
            cell_bg_rect = (x_pos - 5, y_pos - 5, 
                           x_pos + frame_width + 5, y_pos + frame_height + 5)
            draw.rectangle(cell_bg_rect, fill=(230, 230, 230))
            
            # Paste the frame
            composite.paste(std_frame, (x_pos, y_pos))
            
            # Add label
            label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                            x_pos + frame_width, y_pos + frame_height)
            draw.rectangle(label_bg_rect, fill=(0, 0, 0, 180))
            draw_centered_text(draw, "Standard Crop", x_pos + frame_width // 2, 
                              y_pos + frame_height - 15, label_font)
        
        # Add standard crop - bottom row (normalized)
        if 'standard_normalized' in gif_frames and i < len(gif_frames['standard_normalized']):
            # Get the frame
            std_norm_frame = gif_frames['standard_normalized'][i].copy()
            
            # Add border
            std_norm_frame = add_colored_border(std_norm_frame, (0, 0, 0), width=3)
            
            # Calculate position
            x_pos = margin
            y_pos = norm_start_y
            
            # Add background for cell
            cell_bg_rect = (x_pos - 5, y_pos - 5, 
                           x_pos + frame_width + 5, y_pos + frame_height + 5)
            draw.rectangle(cell_bg_rect, fill=(230, 230, 230))
            
            # Paste the frame
            composite.paste(std_norm_frame, (x_pos, y_pos))
            
            # Add label
            label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                            x_pos + frame_width, y_pos + frame_height)
            draw.rectangle(label_bg_rect, fill=(0, 0, 0, 180))
            draw_centered_text(draw, "Standard Crop (Normalized)", x_pos + frame_width // 2, 
                              y_pos + frame_height - 15, label_font)
        
        # Add intelligent crops with different weights - both rows
        for idx, weight in enumerate(weights):
            # Calculate positions
            x_pos = margin + (idx + 1) * (frame_width + margin)
            
            # Colors for this weight
            weight_color = weight_colors[weight]
            weight_bg = weight_bg_colors[weight]
            
            # Add background coloring for column
            col_bg_rect = (x_pos - 5, title_height + header_height, 
                          x_pos + frame_width + 5, composite_height - footer_height)
            draw.rectangle(col_bg_rect, fill=(weight_bg[0], weight_bg[1], weight_bg[2], 50))
            
            # Process unnormalized intelligent crop
            key = f"intelligent_w{weight}"
            if key in gif_frames and i < len(gif_frames[key]):
                # Get the frame and add colored border
                intel_frame = gif_frames[key][i].copy()
                intel_frame = add_colored_border(intel_frame, weight_color, width=5)
                
                # Add background for cell
                cell_bg_rect = (x_pos - 5, unnorm_start_y - 5, 
                               x_pos + frame_width + 5, unnorm_start_y + frame_height + 5)
                draw.rectangle(cell_bg_rect, fill=weight_bg)
                
                # Paste frame
                y_pos = unnorm_start_y
                composite.paste(intel_frame, (x_pos, y_pos))
                
                # Add label with weight color
                label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                                x_pos + frame_width, y_pos + frame_height)
                draw.rectangle(label_bg_rect, fill=(weight_color[0], weight_color[1], weight_color[2], 180))
                draw_centered_text(draw, f"Weight: {weight}", x_pos + frame_width // 2, 
                                  y_pos + frame_height - 15, label_font, fill=(255, 255, 255))
            
            # Process normalized intelligent crop
            key = f"intelligent_w{weight}_normalized"
            if key in gif_frames and i < len(gif_frames[key]):
                # Get the frame and add colored border
                intel_norm_frame = gif_frames[key][i].copy()
                intel_norm_frame = add_colored_border(intel_norm_frame, weight_color, width=5)
                
                # Add background for cell
                cell_bg_rect = (x_pos - 5, norm_start_y - 5, 
                               x_pos + frame_width + 5, norm_start_y + frame_height + 5)
                draw.rectangle(cell_bg_rect, fill=weight_bg)
                
                # Paste frame
                y_pos = norm_start_y
                composite.paste(intel_norm_frame, (x_pos, y_pos))
                
                # Add label with weight color
                label_bg_rect = (x_pos, y_pos + frame_height - 30, 
                                x_pos + frame_width, y_pos + frame_height)
                draw.rectangle(label_bg_rect, fill=(weight_color[0], weight_color[1], weight_color[2], 180))
                draw_centered_text(draw, f"Weight: {weight} (Norm)", x_pos + frame_width // 2, 
                                  y_pos + frame_height - 15, label_font, fill=(255, 255, 255))
        
        # Add frame counter and metadata at bottom
        footer_rect = (0, composite_height - footer_height, composite_width, composite_height)
        draw.rectangle(footer_rect, fill=(50, 50, 50))
        
        # Add frame counter
        frame_text = f"Frame {i+1}/{num_frames}"
        draw_centered_text(draw, frame_text, 100, composite_height - footer_height//2, small_font)
        
        # Add legend
        legend_text = "Red = 0.3 | Blue = 0.5 | Green = 0.7"
        draw_centered_text(draw, legend_text, composite_width - 200, 
                          composite_height - footer_height//2, small_font)
        
        composite_frames.append(composite)
    
    # Save the composite GIF
    output_path = os.path.join(save_gifs_dir, f"{sample_key}_comprehensive.gif")
    logger.info(f"Saving comprehensive comparison to {output_path} with {len(composite_frames)} frames")
    try:
        composite_frames[0].save(
            output_path,
            save_all=True,
            append_images=composite_frames[1:],
            optimize=False,
            duration=200,
            loop=0
        )
        logger.info(f"Saved comprehensive comparison to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save comprehensive comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return output_path

def main():
    # Initialize and parse configuration
    config.parse_args()
    
    # Select a few bboxes for demo
    BBOX_NAMES = [ 'bbox6']
    
    # Parameters for intelligent cropping
    presynapse_weights = [0.3, 0.5, 0.7]  # Different weights to compare
    
    # Set segmentation type
    segmentation_type = 10
    
    # Output directory - using a directory within newdl for easier access
    save_gifs_dir = "newdl/crop_comparison"
    os.makedirs(save_gifs_dir, exist_ok=True)
    
    logger.info(f"Output directory: {save_gifs_dir}")
    logger.info(f"Starting with bbox_names: {BBOX_NAMES}")
    logger.info(f"Raw data directory: {config.raw_base_dir}")
    logger.info(f"Segmentation directory: {config.seg_base_dir}")
    logger.info(f"Additional mask directory: {config.add_mask_base_dir}")
    logger.info(f"Excel file directory: {config.excel_file}")

    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    vol_data_dict = {}
    for bbox in BBOX_NAMES:
        logger.info(f"Loading volumes for {bbox}...")
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox)
        if raw_vol is not None:
            logger.info(f"Successfully loaded volumes for {bbox}")
            vol_data_dict[bbox] = (raw_vol, seg_vol, add_mask_vol)
        else:
            logger.warning(f"Could not load volumes for {bbox}")
    
    # Load synapse data and keep track of available samples
    available_samples = []
    if config.excel_file:
        try:
            logger.info(f"Loading Excel files from {config.excel_file}")
            syn_df_list = []
            
            for bbox in BBOX_NAMES:
                excel_path = os.path.join(config.excel_file, f"{bbox}.xlsx")
                if os.path.exists(excel_path):
                    logger.info(f"Loading {excel_path}")
                    bbox_df = pd.read_excel(excel_path).assign(bbox_name=bbox)
                    syn_df_list.append(bbox_df)
                    
                    # Get available samples from this bbox
                    if bbox in vol_data_dict:  # Only if volumes were loaded successfully
                        # Get a few sample names for each bbox (limit to 2 per bbox to keep output manageable)
                        sample_vars = bbox_df['Var1'].unique()[:2]  
                        logger.info(f"Available samples for {bbox}: {sample_vars}")
                        
                        for var1 in sample_vars:
                            available_samples.append({
                                "bbox_name": bbox,
                                "Var1": var1
                            })
            
            if syn_df_list:
                syn_df = pd.concat(syn_df_list)
                logger.info(f"Loaded synapse data: {len(syn_df)} rows")
            else:
                logger.warning("No data loaded from Excel files")
                syn_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            syn_df = pd.DataFrame()
    else:
        logger.warning("No excel_file specified. Using empty DataFrame.")
        syn_df = pd.DataFrame()
    
    # Add fixed samples if available samples list is empty
    if not available_samples and fixed_samples:
        logger.info("Using predefined fixed samples")
        available_samples = fixed_samples
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Process available samples
    if available_samples:
        logger.info(f"Processing {len(available_samples)} available samples")
        
        # Process all available samples
        for sample in available_samples:
            sample_key = f"{sample['bbox_name']}_{sample['Var1']}"
            logger.info(f"Processing sample: {sample_key}")
            
            # Dictionary to store all frame paths for comprehensive comparison
            all_frame_paths = {}
            
            # FIRST: Generate required standard crops (both normalized and unnormalized)
            logger.info(f"Generating standard crops for {sample_key}...")
            
            # 1. Standard crop (unnormalized)
            try:
                visualize_comparison(
                    syn_df=syn_df,
                    bbox_name=sample['bbox_name'],
                    var1=sample['Var1'],
                    vol_data_dict=vol_data_dict,
                    save_gifs_dir=save_gifs_dir,
                    segmentation_type=segmentation_type,
                    processor=processor,
                    subvol_size=config.subvol_size,
                    num_frames=config.num_frames,
                    alpha=config.alpha,
                    presynapse_weight=0.5,  # Default weight for standard crop
                    normalize_presynapse_size=False
                )
                
                standard_path = os.path.join(save_gifs_dir, f"{sample_key}_standard.gif")
                if os.path.exists(standard_path):
                    logger.info(f"Standard crop created: {standard_path}")
                    all_frame_paths['standard'] = standard_path
                else:
                    logger.error(f"Failed to create standard crop")
            except Exception as e:
                logger.error(f"Error generating standard crop: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 2. Standard crop (normalized)
            try:
                visualize_comparison(
                    syn_df=syn_df,
                    bbox_name=sample['bbox_name'],
                    var1=sample['Var1'],
                    vol_data_dict=vol_data_dict,
                    save_gifs_dir=save_gifs_dir,
                    segmentation_type=segmentation_type,
                    processor=processor,
                    subvol_size=config.subvol_size,
                    num_frames=config.num_frames,
                    alpha=config.alpha,
                    presynapse_weight=0.5,  # Default weight for standard crop
                    normalize_presynapse_size=True,
                    target_percentage=0.15,
                    size_tolerance=0.1
                )
                
                standard_norm_path = os.path.join(save_gifs_dir, f"{sample_key}_standard_normalized.gif")
                if os.path.exists(standard_norm_path):
                    logger.info(f"Normalized standard crop created: {standard_norm_path}")
                    all_frame_paths['standard_normalized'] = standard_norm_path
                else:
                    logger.error(f"Failed to create normalized standard crop")
            except Exception as e:
                logger.error(f"Error generating normalized standard crop: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # SECOND: Generate all intelligent crops with all weights
            for weight in presynapse_weights:
                logger.info(f"Processing {sample_key} with weight {weight}...")
                
                # 1. Unnormalized intelligent crop
                try:
                    visualize_comparison(
                        syn_df=syn_df,
                        bbox_name=sample['bbox_name'],
                        var1=sample['Var1'],
                        vol_data_dict=vol_data_dict,
                        save_gifs_dir=save_gifs_dir,
                        segmentation_type=segmentation_type,
                        processor=processor,
                        subvol_size=config.subvol_size,
                        num_frames=config.num_frames,
                        alpha=config.alpha,
                        presynapse_weight=weight,
                        normalize_presynapse_size=False
                    )
                    
                    intelligent_path = os.path.join(save_gifs_dir, f"{sample_key}_intelligent_w{weight}.gif")
                    if os.path.exists(intelligent_path):
                        logger.info(f"Intelligent crop with weight {weight} created: {intelligent_path}")
                        all_frame_paths[f"intelligent_w{weight}"] = intelligent_path
                    else:
                        logger.error(f"Failed to create intelligent crop with weight {weight}")
                except Exception as e:
                    logger.error(f"Error generating intelligent crop with weight {weight}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 2. Normalized intelligent crop
                try:
                    logger.info(f"Processing {sample_key} with weight {weight} and size normalization...")
                    visualize_comparison(
                        syn_df=syn_df,
                        bbox_name=sample['bbox_name'],
                        var1=sample['Var1'],
                        vol_data_dict=vol_data_dict,
                        save_gifs_dir=save_gifs_dir,
                        segmentation_type=segmentation_type,
                        processor=processor,
                        subvol_size=config.subvol_size,
                        num_frames=config.num_frames,
                        alpha=config.alpha,
                        presynapse_weight=weight,
                        normalize_presynapse_size=True,
                        target_percentage=0.15,
                        size_tolerance=0.1
                    )
                    
                    intelligent_norm_path = os.path.join(save_gifs_dir, f"{sample_key}_intelligent_w{weight}_normalized.gif")
                    if os.path.exists(intelligent_norm_path):
                        logger.info(f"Normalized intelligent crop with weight {weight} created: {intelligent_norm_path}")
                        all_frame_paths[f"intelligent_w{weight}_normalized"] = intelligent_norm_path
                    else:
                        logger.error(f"Failed to create normalized intelligent crop with weight {weight}")
                except Exception as e:
                    logger.error(f"Error generating normalized intelligent crop with weight {weight}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # THIRD: Create all combinations comparison grid
            try:
                logger.info(f"Creating all combinations comparison for {sample_key}...")
                output_path = create_all_combinations_comparison(
                    sample_key=sample_key,
                    frame_paths=all_frame_paths,
                    save_gifs_dir=save_gifs_dir,
                    weights=presynapse_weights
                )
                
                if output_path and os.path.exists(output_path):
                    logger.info(f"Successfully created all combinations comparison: {output_path}")
                else:
                    logger.error("Failed to create all combinations comparison")
            except Exception as e:
                logger.error(f"Error creating all combinations comparison: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # FOURTH: Generate multi-weight comparison and size normalization comparison
            try:
                logger.info(f"Creating multi-weight comparison for {sample_key}...")
                # Build a dictionary of weight frames
                weight_frames = {'standard': all_frame_paths.get('standard')}
                for weight in presynapse_weights:
                    weight_key = f"intelligent_w{weight}"
                    if weight_key in all_frame_paths:
                        weight_frames[weight] = all_frame_paths[weight_key]
                
                create_multi_weight_comparison(
                    sample_key=sample_key,
                    weight_frames=weight_frames,
                    save_gifs_dir=save_gifs_dir,
                    weights=presynapse_weights
                )
            except Exception as e:
                logger.error(f"Error creating multi-weight comparison: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # FIFTH: Generate size normalization comparison for middle weight (0.5)
            try:
                logger.info(f"Creating size normalization comparison for {sample_key}...")
                middle_weight = 0.5
                
                norm_frames = {
                    'standard': all_frame_paths.get('standard'),
                    'standard_normalized': all_frame_paths.get('standard_normalized'),
                    'intelligent': all_frame_paths.get(f'intelligent_w{middle_weight}'),
                    'intelligent_normalized': all_frame_paths.get(f'intelligent_w{middle_weight}_normalized')
                }
                
                create_size_normalization_comparison(
                    sample_key=sample_key,
                    frame_paths=norm_frames,
                    save_gifs_dir=save_gifs_dir,
                    weight=middle_weight
                )
            except Exception as e:
                logger.error(f"Error creating size normalization comparison: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # SIXTH: Check if we have all required frames for comprehensive comparison
            missing_frames = []
            
            # Required frames: standard (normalized and unnormalized) + all weights (normalized and unnormalized)
            required_frames = ['standard', 'standard_normalized']
            for weight in presynapse_weights:
                required_frames.append(f"intelligent_w{weight}")
                required_frames.append(f"intelligent_w{weight}_normalized")
            
            for frame in required_frames:
                if frame not in all_frame_paths:
                    missing_frames.append(frame)
            
            if missing_frames:
                logger.warning(f"Missing frames for comprehensive comparison: {missing_frames}")
            else:
                # Create comprehensive comparison
                logger.info(f"Creating comprehensive comparison for {sample_key}...")
                try:
                    output_path = create_comprehensive_comparison(
                        sample_key=sample_key,
                        frame_paths=all_frame_paths,
                        save_gifs_dir=save_gifs_dir,
                        weights=presynapse_weights
                    )
                    
                    if output_path and os.path.exists(output_path):
                        logger.info(f"Successfully created comprehensive comparison: {output_path}")
                    else:
                        logger.error("Failed to create comprehensive comparison")
                except Exception as e:
                    logger.error(f"Error creating comprehensive comparison: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    else:
        logger.warning("No samples available for processing")
    
    logger.info("Visualization complete")

if __name__ == "__main__":
    main() 