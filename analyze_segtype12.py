import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import torch
import logging
from tqdm import tqdm

# Add the necessary imports from the project
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset2 import SynapseDataset
from synapse.utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segtype12_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("segtype12_analysis")

class Segtype12Analyzer:
    """
    Class for analyzing statistics of segmentation type 12 (vesicle cloud).
    
    This analyzer loads data with segmentation type 12 and computes various
    statistics about discarded samples, vesicle cloud portions, etc.
    """
    
    def __init__(self, bbox_names=None, vesicle_fill_thresholds=None, output_dir="results/segtype12_analysis"):
        """
        Initialize the Segtype12Analyzer.
        
        Args:
            bbox_names (list): List of bounding box names to analyze. 
                              If None, all available bboxes will be used.
            vesicle_fill_thresholds (list): List of vesicle fill thresholds to test.
                                           If None, [25, 50, 75, 90, 95, 99] will be used.
            output_dir (str): Directory to save analysis results.
        """
        # Initialize configuration
        config.parse_args()
        
        # Set default bbox_names if not provided
        self.bbox_names = bbox_names if bbox_names is not None else [
            'bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'
        ]
        
        # Set default vesicle_fill_thresholds if not provided
        self.vesicle_fill_thresholds = vesicle_fill_thresholds if vesicle_fill_thresholds is not None else [
            25.0, 50.0, 75.0, 90.0, 95.0, 99.0
        ]
        
        # Fixed segmentation type for this analyzer
        self.segmentation_type = 12
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures for results
        self.results = {
            'discarded_samples': defaultdict(list),
            'vesicle_portions': defaultdict(list),
            'bbox_stats': defaultdict(dict),
            'threshold_stats': defaultdict(dict),
            'sample_stats': defaultdict(dict),  # New: Store sample-level statistics
        }
        
        # Initialize dataloader and processor
        self.data_loader = SynapseDataLoader(
            raw_base_dir=config.raw_base_dir,
            seg_base_dir=config.seg_base_dir,
            add_mask_base_dir=config.add_mask_base_dir
        )
        
        # Initialize processor
        self.processor = Synapse3DProcessor(size=config.size)
        
        # Load volume data
        self.vol_data_dict = self._load_volumes()
        
        # Load synapse data
        self.synapse_df = self._load_synapse_data()
        
    def _load_volumes(self):
        """Load volumes for all specified bounding boxes."""
        vol_data_dict = {}
        for bbox in self.bbox_names:
            logger.info(f"Loading volumes for {bbox}...")
            raw_vol, seg_vol, add_mask_vol = self.data_loader.load_volumes(bbox)
            if raw_vol is not None:
                logger.info(f"Successfully loaded volumes for {bbox}")
                logger.info(f"Raw volume shape: {raw_vol.shape}")
                logger.info(f"Seg volume shape: {seg_vol.shape}")
                if add_mask_vol is not None:
                    logger.info(f"Add mask volume shape: {add_mask_vol.shape}")
                vol_data_dict[bbox] = (raw_vol, seg_vol, add_mask_vol)
            else:
                logger.warning(f"Could not load volumes for {bbox}")
        
        return vol_data_dict
    
    def _load_synapse_data(self):
        """Load synapse data from Excel files."""
        if config.excel_file:
            try:
                logger.info(f"Loading Excel files from {config.excel_file}")
                excel_files = [f"{bbox}.xlsx" for bbox in self.bbox_names]
                available_excel_files = [f for f in excel_files if os.path.exists(os.path.join(config.excel_file, f))]
                logger.info(f"Available Excel files: {available_excel_files}")
                
                syn_df = pd.concat([
                    pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
                    for bbox in self.bbox_names if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
                ])
                logger.info(f"Loaded synapse data: {len(syn_df)} rows")
                
                if not syn_df.empty:
                    logger.info(f"Sample columns: {syn_df.columns.tolist()}")
                    logger.info(f"Sample data:\n{syn_df.head()}")
                    
                    # Count samples per bbox
                    bbox_counts = syn_df['bbox_name'].value_counts()
                    logger.info(f"Samples per bbox:\n{bbox_counts}")
                    
                    return syn_df
                else:
                    logger.warning(f"No data found in Excel files for the specified bboxes")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading Excel files: {e}")
                return pd.DataFrame()
        else:
            logger.warning("No excel_file specified. Using empty DataFrame.")
            return pd.DataFrame()
    
    def run_analysis(self):
        """Run the analysis for all thresholds and bounding boxes."""
        if self.synapse_df.empty:
            logger.error("No synapse data available. Cannot run analysis.")
            return
        
        # Create dataset for each threshold and compute statistics
        for threshold in self.vesicle_fill_thresholds:
            logger.info(f"Analyzing with vesicle_fill_threshold={threshold}...")
            
            # Create dataset with current threshold
            dataset = SynapseDataset(
                vol_data_dict=self.vol_data_dict,
                synapse_df=self.synapse_df,
                processor=self.processor,
                segmentation_type=self.segmentation_type,
                subvol_size=config.subvol_size,
                num_frames=config.num_frames,
                alpha=config.alpha,
                vesicle_fill_threshold=threshold
            )
            
            # Process each sample and collect statistics
            discarded_samples = []
            vesicle_portions = []
            bbox_discarded = defaultdict(int)
            bbox_total = defaultdict(int)
            
            # Track sample-level data
            sample_data = []
            
            for idx in tqdm(range(len(dataset)), desc=f"Processing samples (threshold={threshold})"):
                sample_data_tensor, syn_info, bbox_name = dataset[idx]
                
                # Extract relevant sample info
                sample_id = syn_info.get('Var1', f'sample_{idx}')
                
                # Check for tensor data issues
                is_all_zeros = torch.all(sample_data_tensor == 0).item()
                
                # We need to actually calculate the vesicle cloud information directly 
                # from the raw data since the dataset may have already discarded it
                
                # Get coordinates for this sample
                try:
                    central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
                    side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
                    side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
                    
                    # Get the raw volumes for this bbox
                    raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
                    
                    # Check if volumes are available
                    if raw_vol is not None and add_mask_vol is not None:
                        # Create a simpler data loader just for analysis
                        analysis_loader = SynapseDataLoader("", "", "")
                        
                        # Get the vesicle cloud information (without rendering)
                        # This will give us the real vesicle cloud data even if the sample was discarded
                        vesicle_info = analysis_loader.get_vesicle_cloud_info(
                            raw_vol=raw_vol,
                            seg_vol=seg_vol,
                            add_mask_vol=add_mask_vol,
                            central_coord=central_coord,
                            side1_coord=side1_coord,
                            side2_coord=side2_coord,
                            subvolume_size=config.subvol_size,
                            bbox_name=bbox_name
                        )
                        
                        if vesicle_info:
                            # Extract vesicle cloud information
                            vesicle_pixel_count = vesicle_info.get('vesicle_pixel_count', 0)
                            total_subvol_pixels = vesicle_info.get('total_subvol_pixels', 0)
                            vesicle_portion = vesicle_info.get('vesicle_portion', 0)
                            vesicle_portion_percent = vesicle_portion * 100
                            meets_threshold = vesicle_portion_percent >= threshold
                            
                            # Get detailed discard information
                            has_enough_pixels = vesicle_info.get('has_enough_vesicle_pixels', False)
                            best_box_fill_percentage = vesicle_info.get('best_box_fill_percentage', 0.0)
                            discard_reason = vesicle_info.get('discard_reason', None)
                            
                            # Store the real vesicle cloud information
                            sample_info = {
                                'sample_id': sample_id,
                                'bbox_name': bbox_name,
                                'vesicle_pixel_count': vesicle_pixel_count,
                                'total_pixels': total_subvol_pixels,
                                'vesicle_portion': vesicle_portion,
                                'vesicle_portion_percent': vesicle_portion_percent,
                                'meets_threshold': bool(meets_threshold),
                                'is_discarded': bool(is_all_zeros),
                                'has_enough_pixels': has_enough_pixels,
                                'best_box_fill_percentage': best_box_fill_percentage,
                                'discard_reason': discard_reason
                            }
                            sample_data.append(sample_info)
                            
                            # Update tracking variables
                            if is_all_zeros:
                                discarded_samples.append(idx)
                                bbox_discarded[bbox_name] += 1
                            else:
                                vesicle_portions.append(vesicle_portion)
                        else:
                            # If vesicle_info is None, we couldn't get the data
                            sample_info = {
                                'sample_id': sample_id,
                                'bbox_name': bbox_name,
                                'vesicle_pixel_count': 0,
                                'total_pixels': sample_data_tensor.numel(),
                                'vesicle_portion': 0,
                                'vesicle_portion_percent': 0,
                                'meets_threshold': False,
                                'is_discarded': True,
                                'has_enough_pixels': False,
                                'best_box_fill_percentage': 0.0,
                                'discard_reason': "Failed to extract vesicle cloud information"
                            }
                            sample_data.append(sample_info)
                            discarded_samples.append(idx)
                            bbox_discarded[bbox_name] += 1
                    else:
                        # Fall back to normal tensor-based analysis if volumes aren't available
                        non_zero_count = torch.count_nonzero(sample_data_tensor).item()
                        total_pixels = sample_data_tensor.numel()
                        vesicle_portion = (non_zero_count / total_pixels) if total_pixels > 0 else 0
                        vesicle_portion_percent = vesicle_portion * 100
                        meets_threshold = vesicle_portion_percent >= threshold
                        
                        rejection_reason = None
                        if is_all_zeros:
                            rejection_reason = "Volumes not available for detailed analysis"
                        
                        sample_info = {
                            'sample_id': sample_id,
                            'bbox_name': bbox_name,
                            'vesicle_pixel_count': non_zero_count,
                            'total_pixels': total_pixels,
                            'vesicle_portion': vesicle_portion,
                            'vesicle_portion_percent': vesicle_portion_percent,
                            'meets_threshold': bool(meets_threshold),
                            'is_discarded': bool(is_all_zeros),
                            'has_enough_pixels': False,
                            'best_box_fill_percentage': 0.0,
                            'discard_reason': rejection_reason
                        }
                        sample_data.append(sample_info)
                        
                        if is_all_zeros:
                            discarded_samples.append(idx)
                            bbox_discarded[bbox_name] += 1
                        else:
                            vesicle_portions.append(vesicle_portion)
                except Exception as e:
                    logger.error(f"Error processing sample {sample_id} from {bbox_name}: {e}")
                    # Fall back to simpler approach on error
                    non_zero_count = torch.count_nonzero(sample_data_tensor).item()
                    total_pixels = sample_data_tensor.numel()
                    vesicle_portion = (non_zero_count / total_pixels) if total_pixels > 0 else 0
                    meets_threshold = vesicle_portion * 100 >= threshold
                    
                    sample_info = {
                        'sample_id': sample_id,
                        'bbox_name': bbox_name,
                        'vesicle_pixel_count': non_zero_count,
                        'total_pixels': total_pixels,
                        'vesicle_portion': vesicle_portion,
                        'vesicle_portion_percent': vesicle_portion * 100,
                        'meets_threshold': bool(meets_threshold),
                        'is_discarded': bool(is_all_zeros),
                        'has_enough_pixels': False,
                        'best_box_fill_percentage': 0.0,
                        'discard_reason': "Error during analysis"
                    }
                    sample_data.append(sample_info)
                    
                    if is_all_zeros:
                        discarded_samples.append(idx)
                        bbox_discarded[bbox_name] += 1
                    else:
                        vesicle_portions.append(vesicle_portion)
                
                bbox_total[bbox_name] += 1
            
            # Store results for this threshold
            self.results['discarded_samples'][threshold] = discarded_samples
            self.results['vesicle_portions'][threshold] = vesicle_portions
            self.results['sample_stats'][threshold] = sample_data
            
            # Calculate discard rate for each bbox
            for bbox_name in self.bbox_names:
                if bbox_name in bbox_total and bbox_total[bbox_name] > 0:
                    discard_rate = (bbox_discarded[bbox_name] / bbox_total[bbox_name]) * 100.0
                    self.results['bbox_stats'][bbox_name][threshold] = {
                        'total': bbox_total[bbox_name],
                        'discarded': bbox_discarded[bbox_name],
                        'discard_rate': discard_rate
                    }
            
            # Calculate overall statistics for this threshold
            total_samples = len(dataset)
            total_discarded = len(discarded_samples)
            discard_rate = (total_discarded / total_samples) * 100.0 if total_samples > 0 else 0.0
            
            self.results['threshold_stats'][threshold] = {
                'total_samples': total_samples,
                'discarded_samples': total_discarded,
                'discard_rate': discard_rate,
                'avg_vesicle_portion': np.mean(vesicle_portions) if vesicle_portions else np.nan
            }
            
            logger.info(f"Threshold {threshold}: {discard_rate:.2f}% samples discarded ({total_discarded}/{total_samples})")
        
        # Generate plots and reports
        self._generate_plots()
        self._generate_sample_plots()  # New: Generate sample-level plots
        self._generate_report()
    
    def _generate_plots(self):
        """Generate plots to visualize the analysis results."""
        # 1. Plot discard rates by threshold
        thresholds = sorted(self.results['threshold_stats'].keys())
        discard_rates = [self.results['threshold_stats'][t]['discard_rate'] for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, discard_rates, 'o-', linewidth=2)
        plt.xlabel('Vesicle Fill Threshold (%)')
        plt.ylabel('Discard Rate (%)')
        plt.title('Sample Discard Rate by Vesicle Fill Threshold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'discard_rate_by_threshold.png'), dpi=300)
        plt.close()
        
        # 2. Plot discard rates by bbox for each threshold
        plt.figure(figsize=(12, 8))
        
        for threshold in thresholds:
            bbox_names = []
            bbox_discard_rates = []
            
            for bbox_name in self.bbox_names:
                if bbox_name in self.results['bbox_stats'] and threshold in self.results['bbox_stats'][bbox_name]:
                    bbox_names.append(bbox_name)
                    bbox_discard_rates.append(self.results['bbox_stats'][bbox_name][threshold]['discard_rate'])
            
            if bbox_names:
                plt.plot(bbox_names, bbox_discard_rates, 'o-', linewidth=2, label=f'Threshold={threshold}%')
        
        plt.xlabel('Bounding Box')
        plt.ylabel('Discard Rate (%)')
        plt.title('Sample Discard Rate by Bounding Box and Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'discard_rate_by_bbox.png'), dpi=300)
        plt.close()
        
        # 3. Plot vesicle portion distribution for the middle threshold
        if self.vesicle_fill_thresholds:
            mid_threshold = self.vesicle_fill_thresholds[len(self.vesicle_fill_thresholds)//2]
            vesicle_portions = self.results['vesicle_portions'][mid_threshold]
            
            if vesicle_portions:
                plt.figure(figsize=(10, 6))
                sns.histplot(vesicle_portions, bins=20, kde=True)
                plt.xlabel('Vesicle Portion (Non-zero ratio)')
                plt.ylabel('Count')
                plt.title(f'Distribution of Vesicle Cloud Portions (Threshold={mid_threshold}%)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'vesicle_portions_threshold_{mid_threshold}.png'), dpi=300)
                plt.close()
        
        # 4. Plot vesicle portions vs threshold
        avg_portions = [self.results['threshold_stats'][t].get('avg_vesicle_portion', np.nan) for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, avg_portions, 'o-', linewidth=2)
        plt.xlabel('Vesicle Fill Threshold (%)')
        plt.ylabel('Average Vesicle Portion')
        plt.title('Average Vesicle Cloud Portion by Threshold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_vesicle_portion_by_threshold.png'), dpi=300)
        plt.close()
        
        # 5. Create heatmap of discard rates by bbox and threshold
        discard_rates_data = []
        for bbox_name in self.bbox_names:
            row = []
            for threshold in thresholds:
                if bbox_name in self.results['bbox_stats'] and threshold in self.results['bbox_stats'][bbox_name]:
                    row.append(self.results['bbox_stats'][bbox_name][threshold]['discard_rate'])
                else:
                    row.append(np.nan)
            discard_rates_data.append(row)
        
        discard_rates_df = pd.DataFrame(discard_rates_data, index=self.bbox_names, columns=thresholds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(discard_rates_df, annot=True, cmap='YlOrRd', fmt='.1f', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Discard Rate (%)'})
        plt.xlabel('Vesicle Fill Threshold (%)')
        plt.ylabel('Bounding Box')
        plt.title('Discard Rate (%) by Bounding Box and Threshold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'discard_rate_heatmap.png'), dpi=300)
        plt.close()
    
    def _generate_sample_plots(self):
        """Generate detailed sample-level plots."""
        # Create a subdirectory for sample plots
        sample_plots_dir = os.path.join(self.output_dir, "sample_plots")
        os.makedirs(sample_plots_dir, exist_ok=True)
        
        # For each threshold, create detailed sample-level plots
        for threshold in self.vesicle_fill_thresholds:
            if threshold not in self.results['sample_stats']:
                continue
                
            sample_data = self.results['sample_stats'][threshold]
            
            # Convert to DataFrame for easier plotting
            sample_df = pd.DataFrame(sample_data)
            
            # Skip if no data
            if sample_df.empty:
                continue
                
            # 1. Plot vesicle pixel counts by sample
            plt.figure(figsize=(14, 8))
            
            # Sort by vesicle pixel count
            sorted_df = sample_df.sort_values('vesicle_pixel_count', ascending=False)
            
            # Color by whether the sample meets the threshold
            colors = ['green' if meets else 'red' for meets in sorted_df['meets_threshold']]
            
            # Plot bars
            plt.bar(range(len(sorted_df)), sorted_df['vesicle_pixel_count'], color=colors)
            
            # Add a horizontal line at the threshold level (if we can calculate it)
            if not sample_df.empty and 'total_pixels' in sample_df:
                avg_total_pixels = sample_df['total_pixels'].mean()
                threshold_pixel_count = avg_total_pixels * (threshold / 100)
                plt.axhline(y=threshold_pixel_count, color='black', linestyle='--', 
                           label=f'Threshold ({threshold}% of avg total pixels)')
                plt.legend()
            
            plt.xlabel('Sample Index (sorted by pixel count)')
            plt.ylabel('Vesicle Cloud Pixel Count')
            plt.title(f'Vesicle Cloud Pixel Count by Sample (Threshold={threshold}%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add count of samples meeting/not meeting threshold
            meets_count = sorted_df['meets_threshold'].sum()
            fails_count = len(sorted_df) - meets_count
            plt.figtext(0.5, 0.01, 
                      f"Meets threshold: {meets_count} ({meets_count/len(sorted_df)*100:.1f}%), "
                      f"Fails threshold: {fails_count} ({fails_count/len(sorted_df)*100:.1f}%)",
                      ha='center', fontsize=12)
            
            plt.savefig(os.path.join(sample_plots_dir, f'vesicle_pixel_count_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Scatter plot of vesicle portion vs total pixels
            plt.figure(figsize=(12, 8))
            
            # Color points by whether they meet the threshold
            scatter = plt.scatter(sample_df['total_pixels'], sample_df['vesicle_portion_percent'],
                               c=sample_df['meets_threshold'].map({True: 'green', False: 'red'}),
                               alpha=0.7)
            
            # Add a horizontal line at the threshold
            plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold={threshold}%')
            
            # Add a legend
            plt.legend(['Threshold', 'Meets Threshold', 'Fails Threshold'])
            
            plt.xlabel('Total Pixels in Sample')
            plt.ylabel('Vesicle Portion (%)')
            plt.title(f'Vesicle Portion vs Total Pixels (Threshold={threshold}%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_plots_dir, f'vesicle_portion_vs_total_pixels_threshold_{threshold}.png'), dpi=300)
            plt.close()
            
            # 3. Box plot of vesicle portions by bounding box
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='bbox_name', y='vesicle_portion_percent', data=sample_df)
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}%')
            plt.legend()
            plt.xlabel('Bounding Box')
            plt.ylabel('Vesicle Portion (%)')
            plt.title(f'Distribution of Vesicle Portions by Bounding Box (Threshold={threshold}%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_plots_dir, f'vesicle_portion_by_bbox_threshold_{threshold}.png'), dpi=300)
            plt.close()
            
            # 4. NEW: Plot rejection reasons for discarded samples
            if 'discard_reason' in sample_df.columns:
                # Filter to only discarded samples
                discarded_df = sample_df[sample_df['is_discarded'] == True]
                
                if not discarded_df.empty:
                    # Count rejection reasons
                    reason_counts = discarded_df['discard_reason'].value_counts()
                    
                    plt.figure(figsize=(12, 8))
                    reason_counts.plot(kind='bar')
                    plt.xlabel('Rejection Reason')
                    plt.ylabel('Number of Samples')
                    plt.title(f'Reasons for Sample Rejection (Threshold={threshold}%)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_plots_dir, f'rejection_reasons_threshold_{threshold}.png'), dpi=300)
                    plt.close()
                    
                    # Create a detailed table of discarded samples
                    discarded_details_path = os.path.join(sample_plots_dir, f'discarded_samples_details_threshold_{threshold}.csv')
                    discarded_df.to_csv(discarded_details_path, index=False)
                    
                    # Plot histogram of vesicle portions for discarded samples that had some vesicles
                    non_zero_discarded = discarded_df[discarded_df['vesicle_pixel_count'] > 0]
                    if not non_zero_discarded.empty:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(non_zero_discarded['vesicle_portion_percent'], bins=20, kde=True)
                        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold}%')
                        plt.legend()
                        plt.xlabel('Vesicle Portion (%)')
                        plt.ylabel('Number of Samples')
                        plt.title(f'Distribution of Vesicle Portions for Discarded Samples (Threshold={threshold}%)')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(sample_plots_dir, f'discarded_vesicle_portions_threshold_{threshold}.png'), dpi=300)
                        plt.close()
                        
                    # 5. NEW: Plot detailed vesicle pixel count distribution
                    plt.figure(figsize=(12, 8))
                    plt.hist([
                        sample_df[sample_df['is_discarded'] == True]['vesicle_pixel_count'],
                        sample_df[sample_df['is_discarded'] == False]['vesicle_pixel_count']
                    ], bins=30, label=['Discarded Samples', 'Kept Samples'], alpha=0.7)
                    box_size = 25
                    required_pixels = box_size * box_size * box_size  # 15,625 for a 25×25×25 box
                    plt.axvline(x=required_pixels, color='red', linestyle='--', 
                               label=f'Required pixels for {box_size}×{box_size}×{box_size} box ({required_pixels})')
                    plt.xlabel('Vesicle Pixel Count')
                    plt.ylabel('Number of Samples')
                    plt.title(f'Distribution of Vesicle Pixel Counts (Threshold={threshold}%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_plots_dir, f'vesicle_pixel_count_distribution_{threshold}.png'), dpi=300)
                    plt.close()
                    
                    # 6. NEW: Plot best box fill percentage for samples with enough pixels
                    samples_with_enough_pixels = sample_df[sample_df['has_enough_pixels'] == True]
                    if not samples_with_enough_pixels.empty:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(samples_with_enough_pixels['best_box_fill_percentage'], bins=20, kde=True)
                        plt.axvline(x=99.0, color='red', linestyle='--', label=f'Required fill percentage (99.0%)')
                        plt.axvline(x=threshold, color='green', linestyle='--', label=f'Current threshold={threshold}%')
                        plt.legend()
                        plt.xlabel('Best Box Fill Percentage (%)')
                        plt.ylabel('Number of Samples')
                        plt.title(f'Distribution of Best Box Fill Percentages (Threshold={threshold}%)')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(sample_plots_dir, f'best_box_fill_percentages_{threshold}.png'), dpi=300)
                        plt.close()
                        
                    # 7. NEW: Create detailed breakdown of rejection reasons by category
                    if not discarded_df.empty and 'discard_reason' in discarded_df.columns:
                        # Categorize rejection reasons
                        reason_categories = {
                            'No vesicle pixels': discarded_df['discard_reason'].str.contains('No vesicle', case=False, na=False).sum(),
                            'Too few vesicle pixels': discarded_df['discard_reason'].str.contains('too small|<', case=False, na=False).sum(),
                            'Dimensions too small': discarded_df['discard_reason'].str.contains('dimensions', case=False, na=False).sum(),
                            'Insufficient fill percentage': discarded_df['discard_reason'].str.contains('fill percentage', case=False, na=False).sum(),
                            'Other/Error': len(discarded_df) - (
                                discarded_df['discard_reason'].str.contains('No vesicle|too small|<|dimensions|fill percentage', 
                                                                         case=False, na=False).sum()
                            )
                        }
                        
                        # Create pie chart of rejection reason categories
                        plt.figure(figsize=(10, 10))
                        plt.pie(
                            reason_categories.values(), 
                            labels=[f"{k} ({v})" for k, v in reason_categories.items()],
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                        )
                        plt.axis('equal')
                        plt.title(f'Rejection Reason Categories (Threshold={threshold}%)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(sample_plots_dir, f'rejection_reason_categories_{threshold}.png'), dpi=300)
                        plt.close()
                        
                        # 8. NEW: Create scatter plot of vesicle pixels vs best box fill percentage
                        plt.figure(figsize=(12, 8))
                        samples_with_fill = sample_df[~sample_df['best_box_fill_percentage'].isna()]
                        if not samples_with_fill.empty:
                            plt.scatter(
                                samples_with_fill['vesicle_pixel_count'], 
                                samples_with_fill['best_box_fill_percentage'],
                                c=samples_with_fill['is_discarded'].map({True: 'red', False: 'green'}),
                                alpha=0.7
                            )
                            plt.axhline(y=99.0, color='red', linestyle='--', label='Required fill percentage (99.0%)')
                            plt.axvline(x=required_pixels, color='blue', linestyle='--', 
                                      label=f'Required pixels ({required_pixels})')
                            plt.xlabel('Vesicle Pixel Count')
                            plt.ylabel('Best Box Fill Percentage (%)')
                            plt.title(f'Vesicle Pixels vs Box Fill Percentage (Threshold={threshold}%)')
                            plt.legend(['Required fill (99.0%)', 'Required pixels', 'Discarded', 'Kept'])
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(os.path.join(sample_plots_dir, f'pixels_vs_fill_percentage_{threshold}.png'), dpi=300)
                            plt.close()
            
            # 9. Combine detailed sample data into a single CSV for this threshold
            sample_df.to_csv(os.path.join(sample_plots_dir, f'sample_details_threshold_{threshold}.csv'), index=False)
    
    def _generate_report(self):
        """Generate a comprehensive report of the analysis results."""
        # 1. Generate the threshold summary report
        report_path = os.path.join(self.output_dir, 'segtype12_analysis_report.csv')
        
        # Create a DataFrame for the report
        report_data = []
        
        # Add threshold-level statistics
        for threshold in sorted(self.results['threshold_stats'].keys()):
            stats = self.results['threshold_stats'][threshold]
            row = {
                'threshold': threshold,
                'bbox_name': 'ALL',
                'total_samples': stats['total_samples'],
                'discarded_samples': stats['discarded_samples'],
                'discard_rate': stats['discard_rate'],
                'avg_vesicle_portion': stats.get('avg_vesicle_portion', np.nan)
            }
            report_data.append(row)
        
        # Add bbox-level statistics for each threshold
        for threshold in sorted(self.results['threshold_stats'].keys()):
            for bbox_name in self.bbox_names:
                if bbox_name in self.results['bbox_stats'] and threshold in self.results['bbox_stats'][bbox_name]:
                    stats = self.results['bbox_stats'][bbox_name][threshold]
                    row = {
                        'threshold': threshold,
                        'bbox_name': bbox_name,
                        'total_samples': stats['total'],
                        'discarded_samples': stats['discarded'],
                        'discard_rate': stats['discard_rate'],
                        'avg_vesicle_portion': np.nan  # We don't track this per bbox
                    }
                    report_data.append(row)
        
        # Create and save the DataFrame
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_path, index=False)
        logger.info(f"Analysis report saved to {report_path}")
        
        # 2. Generate sample-level detailed report
        sample_report_path = os.path.join(self.output_dir, 'segtype12_sample_report.csv')
        
        # Combine all sample data across thresholds
        all_sample_data = []
        for threshold, samples in self.results['sample_stats'].items():
            for sample in samples:
                sample_copy = sample.copy()
                sample_copy['threshold'] = threshold
                all_sample_data.append(sample_copy)
        
        if all_sample_data:
            sample_report_df = pd.DataFrame(all_sample_data)
            sample_report_df.to_csv(sample_report_path, index=False)
            logger.info(f"Sample-level report saved to {sample_report_path}")
        
        # 3. Create a summary text report
        summary_path = os.path.join(self.output_dir, 'segtype12_analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Segmentation Type 12 (Vesicle Cloud) Analysis Summary\n")
            f.write("==================================================\n\n")
            
            f.write("1. Overall Statistics by Threshold\n")
            f.write("--------------------------------\n")
            for threshold in sorted(self.results['threshold_stats'].keys()):
                stats = self.results['threshold_stats'][threshold]
                f.write(f"Threshold {threshold}%:\n")
                f.write(f"  - Total samples: {stats['total_samples']}\n")
                f.write(f"  - Discarded samples: {stats['discarded_samples']}\n")
                f.write(f"  - Discard rate: {stats['discard_rate']:.2f}%\n")
                f.write(f"  - Avg vesicle portion: {stats.get('avg_vesicle_portion', np.nan):.4f}\n\n")
            
            f.write("2. Statistics by Bounding Box\n")
            f.write("---------------------------\n")
            for bbox_name in self.bbox_names:
                if bbox_name in self.results['bbox_stats']:
                    f.write(f"Bounding Box: {bbox_name}\n")
                    for threshold in sorted(self.results['bbox_stats'][bbox_name].keys()):
                        stats = self.results['bbox_stats'][bbox_name][threshold]
                        f.write(f"  Threshold {threshold}%:\n")
                        f.write(f"    - Total samples: {stats['total']}\n")
                        f.write(f"    - Discarded samples: {stats['discarded']}\n")
                        f.write(f"    - Discard rate: {stats['discard_rate']:.2f}%\n")
                    f.write("\n")
            
            # 3. Add sample-level statistics summary
            f.write("3. Sample-Level Statistics Summary\n")
            f.write("--------------------------------\n")
            for threshold in sorted(self.results['sample_stats'].keys()):
                samples = self.results['sample_stats'][threshold]
                if not samples:
                    continue
                    
                # Count samples that meet/don't meet threshold
                meets_threshold = sum(1 for s in samples if s.get('meets_threshold', False))
                fails_threshold = len(samples) - meets_threshold
                
                f.write(f"Threshold {threshold}%:\n")
                f.write(f"  - Samples meeting threshold: {meets_threshold} ({meets_threshold/len(samples)*100:.1f}%)\n")
                f.write(f"  - Samples failing threshold: {fails_threshold} ({fails_threshold/len(samples)*100:.1f}%)\n")
                
                # Calculate average vesicle portion
                avg_portion = np.mean([s.get('vesicle_portion', 0) for s in samples]) * 100
                f.write(f"  - Average vesicle portion: {avg_portion:.2f}%\n")
                
                # Add rejection reason summary
                discarded_samples = [s for s in samples if s.get('is_discarded', False)]
                if discarded_samples:
                    f.write(f"\n  Rejection Reasons for Discarded Samples ({len(discarded_samples)}):\n")
                    
                    # Group by rejection reason
                    reason_counts = {}
                    for sample in discarded_samples:
                        reason = sample.get('discard_reason', 'Unknown')
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    
                    # Sort by count and print
                    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    - {reason}: {count} samples ({count/len(discarded_samples)*100:.1f}%)\n")
                        
                    # Group by major categories
                    reasons_by_category = {
                        'No vesicle pixels': 0,
                        'Too few vesicle pixels': 0,
                        'Dimensions too small': 0,
                        'Insufficient fill percentage': 0,
                        'Other/Error': 0
                    }
                    
                    for sample in discarded_samples:
                        reason = sample.get('discard_reason', 'Unknown')
                        if not reason:
                            reasons_by_category['Other/Error'] += 1
                        elif 'No vesicle' in reason:
                            reasons_by_category['No vesicle pixels'] += 1
                        elif 'too small' in reason or '<' in reason:
                            reasons_by_category['Too few vesicle pixels'] += 1
                        elif 'dimensions' in reason:
                            reasons_by_category['Dimensions too small'] += 1
                        elif 'fill percentage' in reason:
                            reasons_by_category['Insufficient fill percentage'] += 1
                        else:
                            reasons_by_category['Other/Error'] += 1
                    
                    # Print category summary
                    f.write(f"\n  Rejection Reasons by Category:\n")
                    for category, count in reasons_by_category.items():
                        if count > 0:
                            f.write(f"    - {category}: {count} samples ({count/len(discarded_samples)*100:.1f}%)\n")
                    
                    # Detailed statistics for samples that had enough vesicle pixels but still failed
                    samples_with_enough_pixels = [s for s in discarded_samples if s.get('has_enough_pixels', False)]
                    if samples_with_enough_pixels:
                        fill_percentages = [s.get('best_box_fill_percentage', 0) for s in samples_with_enough_pixels]
                        f.write(f"\n  Statistics for samples with enough vesicle pixels but still rejected ({len(samples_with_enough_pixels)}):\n")
                        f.write(f"    - Average best box fill percentage: {np.mean(fill_percentages):.2f}%\n")
                        f.write(f"    - Min fill percentage: {np.min(fill_percentages):.2f}%\n")
                        f.write(f"    - Max fill percentage: {np.max(fill_percentages):.2f}%\n")
                    
                    # Average vesicle portion for discarded samples that had vesicles
                    non_zero_discarded = [s for s in discarded_samples if s.get('vesicle_pixel_count', 0) > 0]
                    if non_zero_discarded:
                        avg_discarded_portion = np.mean([s.get('vesicle_portion', 0) for s in non_zero_discarded]) * 100
                        f.write(f"    - Average vesicle portion for discarded samples with vesicles: {avg_discarded_portion:.2f}%\n")
                
                f.write("\n")
            
            f.write("4. Recommended Threshold\n")
            f.write("-----------------------\n")
            
            # Find the threshold with discard rate closest to 20% (arbitrary target)
            target_discard_rate = 20.0
            closest_threshold = None
            min_diff = float('inf')
            
            for threshold, stats in self.results['threshold_stats'].items():
                diff = abs(stats['discard_rate'] - target_discard_rate)
                if diff < min_diff:
                    min_diff = diff
                    closest_threshold = threshold
            
            if closest_threshold is not None:
                f.write(f"Recommended threshold: {closest_threshold}%\n")
                f.write(f"  - Discard rate: {self.results['threshold_stats'][closest_threshold]['discard_rate']:.2f}%\n")
                f.write(f"  - This threshold provides a reasonable balance between keeping most samples\n")
                f.write(f"    while filtering out problematic ones with insufficient vesicle cloud data.\n")
            else:
                f.write("Could not determine a recommended threshold.\n")
                
        logger.info(f"Analysis summary saved to {summary_path}")

def main():
    """Main function to run the segmentation type 12 analysis."""
    # Parse command-line arguments
    config.parse_args()
    
    # Create the analyzer with all bounding boxes
    analyzer = Segtype12Analyzer(
        bbox_names=['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'],
        vesicle_fill_thresholds=[ 99.0],
        output_dir="results/segtype12_analysis"
    )
    
    # Run the analysis
    analyzer.run_analysis()
    
    logger.info("Segmentation type 12 analysis completed successfully.")

if __name__ == "__main__":
    main() 