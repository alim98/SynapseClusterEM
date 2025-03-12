import os
import numpy as np
import pandas as pd
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
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
        
        # Try to use a suitable font with larger size
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
            # Get text size
            try:
                text_width, text_height = draw.textsize(text, font=font)
            except AttributeError:
                # For newer Pillow versions
                text_width, text_height = font.getbbox(text)[2:4]
            
            # Ensure text fits by truncating if necessary
            if text_width > max_width:
                while text_width > max_width and len(text) > 3:
                    text = text[:-1]  # Remove last character
                    try:
                        text_width, text_height = draw.textsize(text + "...", font=font)
                    except AttributeError:
                        text_width, text_height = font.getbbox(text + "...")[2:4]
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
            text_width, text_height = draw.textsize(frame_number_text, font=frame_font)
        except AttributeError:
            text_width, text_height = frame_font.getbbox(frame_number_text)[2:4]
            
        draw.text((total_width - text_width - 10, 10), frame_number_text, fill=(80, 80, 80), font=frame_font)
        
        combined_frames.append(np.array(combined))
    
    return combined_frames

def visualize_comparison(syn_df, bbox_name, var1, vol_data_dict, save_gifs_dir, segmentation_type, 
                         processor, subvol_size=80, num_frames=80, alpha=0.3, presynapse_weight=0.5):
    """
    Visualize a specific sample with both standard and intelligent cropping
    """
    logger.info(f"Starting comparison visualization for {var1} from {bbox_name}")
    
    # Filter data based on var1 and bbox_name values
    specific_sample = syn_df[(syn_df['Var1'] == var1) & (syn_df['bbox_name'] == bbox_name)]
    
    # Check if a sample was found
    if specific_sample.empty:
        logger.warning(f"No sample found in syn_df with Var1={var1} and bbox_name={bbox_name}")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(save_gifs_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = SynapseDataLoader("", "", "")
    
    # Get info for the first matching sample
    sample_info = specific_sample.iloc[0]
    
    # Get volumes
    raw_vol, seg_vol, add_mask_vol = vol_data_dict.get(bbox_name, (None, None, None))
    if raw_vol is None:
        logger.warning(f"No volume data found for {bbox_name}")
        return
    
    # Extract coordinates
    central_coord = (int(sample_info['central_coord_1']), int(sample_info['central_coord_2']), int(sample_info['central_coord_3']))
    side1_coord = (int(sample_info['side_1_coord_1']), int(sample_info['side_1_coord_2']), int(sample_info['side_1_coord_3']))
    side2_coord = (int(sample_info['side_2_coord_1']), int(sample_info['side_2_coord_2']), int(sample_info['side_2_coord_3']))
    
    logger.info(f"Creating standard cube for {var1}...")
    # Create standard cube (no smart cropping)
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
        smart_crop=False
    )
    
    logger.info(f"Creating intelligently cropped cube for {var1}...")
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
        presynapse_weight=presynapse_weight
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
    standard_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_standard.gif")
    intelligent_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_intelligent_w{presynapse_weight}.gif")
    
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
    title_intelligent = f"Intelligent Cropping (w={presynapse_weight})"
    if shift_info:
        title_intelligent += f" - Shift: {shift_info}"
    
    # Create and save side-by-side comparison GIF
    combined_frames = create_combined_frames(
        enhanced_standard_frames, 
        enhanced_intelligent_frames,
        title_standard, 
        title_intelligent
    )
    
    combined_gif_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_comparison_w{presynapse_weight}.gif")
    
    try:
        logger.info(f"Saving combined comparison GIF to {combined_gif_path}")
        imageio.mimsave(combined_gif_path, combined_frames, fps=8)  # Slower frame rate
        logger.info("GIF saved successfully")
    except Exception as e:
        logger.error(f"Failed to save combined GIF: {e}")
        
    # Create an info file with details of the comparison
    info_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Sample: {var1} from {bbox_name}\n")
        f.write(f"Segmentation Type: {segmentation_type}\n")
        f.write(f"Presynapse Weight: {presynapse_weight}\n")
        f.write(f"Original Center: {central_coord}\n")
        if shift_info:
            f.write(f"Shift Vector: {shift_info}\n")
        f.write(f"Standard Range: min={standard_min:.4f}, max={standard_max:.4f}\n")
        f.write(f"Intelligent Range: min={intelligent_min:.4f}, max={intelligent_max:.4f}\n")

def main():
    # Initialize and parse configuration
    config.parse_args()
    
    # Select a few bboxes for demo
    BBOX_NAMES = ['bbox3', 'bbox5', 'bbox6']
    
    # Parameters for intelligent cropping
    presynapse_weights = [0.3, 0.5, 0.7]  # Different weights to compare
    
    # Set segmentation type
    segmentation_type = 5
    
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
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Process available samples
    if available_samples:
        logger.info(f"Processing {len(available_samples)} available samples")
        
        # Dictionary to store frames for multi-weight comparison
        multi_weight_frames = {}
        
        # First, generate individual comparisons for each weight
        for sample in available_samples:
            sample_key = f"{sample['bbox_name']}_{sample['Var1']}"
            multi_weight_frames[sample_key] = {}
            
            for weight in presynapse_weights:
                logger.info(f"Processing {sample['Var1']} from {sample['bbox_name']} with weight {weight}...")
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
                    presynapse_weight=weight
                )
                
                # Store the generated GIF path for multi-weight comparison
                gif_path = os.path.join(save_gifs_dir, f"{sample['bbox_name']}_{sample['Var1']}_intelligent_w{weight}.gif")
                standard_gif_path = os.path.join(save_gifs_dir, f"{sample['bbox_name']}_{sample['Var1']}_standard.gif")
                
                if os.path.exists(gif_path):
                    multi_weight_frames[sample_key][weight] = gif_path
                    # Store standard only once
                    if 'standard' not in multi_weight_frames[sample_key] and os.path.exists(standard_gif_path):
                        multi_weight_frames[sample_key]['standard'] = standard_gif_path
            
            # Generate multi-weight comparison GIF if we have at least 2 weights
            if len(multi_weight_frames[sample_key]) > 2:  # Standard + at least 2 weights
                logger.info(f"Creating multi-weight comparison for {sample_key}...")
                create_multi_weight_comparison(
                    sample_key=sample_key,
                    weight_frames=multi_weight_frames[sample_key],
                    save_gifs_dir=save_gifs_dir,
                    weights=presynapse_weights
                )
    else:
        logger.warning("No samples available for processing")
    
    logger.info("Visualization complete")

def create_multi_weight_comparison(sample_key, weight_frames, save_gifs_dir, weights):
    """
    Create a multi-panel comparison of different presynapse weights
    with improved visual styling
    """
    try:
        # Load GIFs
        standard_frames = imageio.mimread(weight_frames['standard'], memtest=False)
        
        weight_gifs = {}
        for weight in weights:
            if weight in weight_frames:
                weight_gifs[weight] = imageio.mimread(weight_frames[weight], memtest=False)
        
        if not weight_gifs:
            logger.warning(f"No weight GIFs found for {sample_key}")
            return
        
        # Check if any frames were loaded
        if not standard_frames or len(standard_frames) == 0:
            logger.warning(f"No frames found in standard GIF for {sample_key}")
            return
        
        # Get frame dimensions
        height, width, _ = standard_frames[0].shape
        
        # Standardize number of frames across GIFs
        min_frames = min(len(standard_frames), *[len(weight_gifs[w]) for w in weight_gifs])
        standard_frames = standard_frames[:min_frames]
        for weight in weight_gifs:
            weight_gifs[weight] = weight_gifs[weight][:min_frames]
        
        # Create multi-panel output frames with improved layout
        multi_frames = []
        margin = 20  # Increased margin
        header_height = 60  # Increased header height
        
        num_panels = 1 + len(weight_gifs)  # Standard + all weights
        total_width = (width * num_panels) + (margin * (num_panels - 1))
        total_height = height + header_height + 10  # Added bottom padding
        
        for i in range(min_frames):
            # Create base image
            combined = Image.new('RGB', (total_width, total_height), color=(240, 240, 240))
            
            # Add header background
            header_bg = Image.new('RGB', (total_width, header_height), color=(230, 230, 230))
            combined.paste(header_bg, (0, 0))
            
            # Add standard frame
            standard_img = Image.fromarray(standard_frames[i])
            combined.paste(standard_img, (0, header_height))
            
            # Add each weight frame
            x_offset = width + margin
            for weight in sorted(weight_gifs.keys()):
                weight_img = Image.fromarray(weight_gifs[weight][i])
                combined.paste(weight_img, (x_offset, header_height))
                x_offset += width + margin
            
            # Add text
            draw = ImageDraw.Draw(combined)
            
            # Try to find a font with larger size
            try:
                title_font = ImageFont.truetype("arial.ttf", 10)
                subtitle_font = ImageFont.truetype("arial.ttf", 8)
                small_font = ImageFont.truetype("arial.ttf", 6)
            except IOError:
                try:
                    title_font = ImageFont.truetype("DejaVuSans.ttf", 10)
                    subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 8)
                    small_font = ImageFont.truetype("DejaVuSans.ttf", 6)
                except IOError:
                    title_font = ImageFont.load_default()
                    subtitle_font = title_font
                    small_font = title_font
            
            # Add main title
            bbox_name, var1 = sample_key.split('_', 1)
            title_text = f"Presynapse Weight Comparison: {var1} ({bbox_name})"
            # Get text width to center it
            try:
                title_width, title_height = draw.textsize(title_text, font=title_font)
            except AttributeError:
                title_width, title_height = title_font.getbbox(title_text)[2:4]
            
            # Draw text with shadow for better visibility
            title_x = (total_width - title_width) // 2
            draw.text((title_x+2, 8+2), title_text, fill=(180, 180, 180), font=title_font)
            draw.text((title_x, 8), title_text, fill=(0, 0, 0), font=title_font)
            
            # Add titles for panels
            # Title for standard
            standard_text = "Standard Cropping"
            try:
                std_width, _ = draw.textsize(standard_text, font=subtitle_font)
            except AttributeError:
                std_width, _ = subtitle_font.getbbox(standard_text)[2:4]
            
            std_x = (width - std_width) // 2
            draw.text((std_x+1, 36+1), standard_text, fill=(180, 180, 180), font=subtitle_font)
            draw.text((std_x, 36), standard_text, fill=(0, 0, 0), font=subtitle_font)
            
            # Titles for weights
            x_offset = width + margin
            for weight in sorted(weight_gifs.keys()):
                weight_title = f"Weight = {weight}"
                try:
                    w_width, _ = draw.textsize(weight_title, font=subtitle_font)
                except AttributeError:
                    w_width, _ = subtitle_font.getbbox(weight_title)[2:4]
                
                w_x = x_offset + (width - w_width) // 2
                draw.text((w_x+1, 36+1), weight_title, fill=(180, 180, 180), font=subtitle_font)
                draw.text((w_x, 36), weight_title, fill=(0, 0, 0), font=subtitle_font)
                x_offset += width + margin
            
            # # Add frame counter
            # frame_text = f"Frame {i+1}/{min_frames}"
            # try:
            #     frame_width, _ = draw.textsize(frame_text, font=small_font)
            # except AttributeError:
            #     frame_width, _ = small_font.getbbox(frame_text)[2:4]
            
            # draw.text((total_width - frame_width - 10, header_height - 25), frame_text, fill=(80, 80, 80), font=small_font)
            
            # Add separator lines with more visibility
            x_pos = width + margin//2
            for _ in range(len(weight_gifs)):
                draw.line([(x_pos, 0), (x_pos, total_height)], fill=(180, 180, 180), width=2)
                x_pos += width + margin
            
            # Add borders around each panel
            for panel in range(num_panels):
                panel_x = panel * (width + margin)
                # Draw rectangle around panel
                draw.rectangle(
                    [(panel_x, header_height), (panel_x + width - 1, total_height - 1)],
                    outline=(150, 150, 150), width=2
                )
            
            # Add to output frames
            multi_frames.append(np.array(combined))
        
        # Save the multi-weight comparison GIF
        output_path = os.path.join(save_gifs_dir, f"{sample_key}_multi_weight_comparison.gif")
        logger.info(f"Saving multi-weight comparison to {output_path}")
        imageio.mimsave(output_path, multi_frames, fps=8)
        logger.info(f"Multi-weight comparison saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating multi-weight comparison for {sample_key}: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 