import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

# Import the necessary classes
from synapse.utils.config import config
from newdl.dataloader3 import SynapseDataLoader
from newdl.dataset3 import SynapseDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segtype12_sanity_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SegType12SanityCheck")

class SegType12SanityCheck:
    """
    Class to perform sanity checks on segmentation type 12 (vesicle cloud with 25×25×25 bounding box).
    Analyzes how many samples are skipped due to not being able to fit a 25×25×25 bounding box.
    """
    
    def __init__(self, bbox_names=None, excel_file=None, raw_base_dir=None, 
                 seg_base_dir=None, add_mask_base_dir=None):
        """
        Initialize the sanity check class.
        
        Args:
            bbox_names: List of bounding box names to check (defaults to config.bbox_name)
            excel_file: Path to excel files (defaults to config.excel_file)
            raw_base_dir: Path to raw data directory (defaults to config.raw_base_dir)
            seg_base_dir: Path to segmentation data directory (defaults to config.seg_base_dir)
            add_mask_base_dir: Path to additional mask directory (defaults to config.add_mask_base_dir)
        """
        self.bbox_names = bbox_names or config.bbox_name
        self.excel_file = excel_file or config.excel_file
        self.raw_base_dir = raw_base_dir or config.raw_base_dir
        self.seg_base_dir = seg_base_dir or config.seg_base_dir
        self.add_mask_base_dir = add_mask_base_dir or config.add_mask_base_dir
        
        # Initialize results dictionary
        self.results = {
            "total_samples": 0,
            "processed_samples": 0,
            "skipped_samples": 0,
            "skip_reasons": {
                "no_vesicle_pixels": 0,
                "not_enough_pixels": 0,
                "dimensions_too_small": 0,
                "cannot_fit_box": 0,
                "other_errors": 0
            },
            "per_bbox": {}
        }
        
        # Initialize data loader
        self.data_loader = SynapseDataLoader(
            raw_base_dir=self.raw_base_dir,
            seg_base_dir=self.seg_base_dir,
            add_mask_base_dir=self.add_mask_base_dir
        )
        
        # Load synapse data
        self.synapse_df = self._load_synapse_data()
        
        # Load volumes
        self.vol_data_dict = self._load_volumes()
    
    def _load_synapse_data(self):
        """Load and combine synapse data from Excel files."""
        try:
            dfs = []
            for bbox in self.bbox_names:
                excel_path = os.path.join(self.excel_file, f"{bbox}.xlsx")
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    df['bbox_name'] = bbox
                    dfs.append(df)
                else:
                    logger.warning(f"Excel file not found: {excel_path}")
            
            if not dfs:
                logger.error("No Excel files were loaded.")
                return pd.DataFrame()
            
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Error loading synapse data: {e}")
            return pd.DataFrame()
    
    def _load_volumes(self):
        """Load all volumes for the specified bounding boxes."""
        vol_data_dict = {}
        for bbox_name in self.bbox_names:
            logger.info(f"Loading volumes for {bbox_name}...")
            raw_vol, seg_vol, add_mask_vol = self.data_loader.load_volumes(bbox_name)
            if raw_vol is not None:
                vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
            else:
                logger.warning(f"Failed to load volumes for {bbox_name}")
        
        return vol_data_dict
    
    def run_check(self, vesicle_fill_threshold=100.0):
        """
        Run the sanity check for segmentation type 12.
        
        Args:
            vesicle_fill_threshold: Required percentage of vesicle fill (default: 100%)
            
        Returns:
            dict: Results of the sanity check
        """
        # Set up a custom logging handler to capture specific log messages
        log_capture_handler = SkipReasonHandler()
        logger.addHandler(log_capture_handler)
        
        # Reset results
        self.results = {
            "total_samples": 0,
            "processed_samples": 0,
            "skipped_samples": 0,
            "skip_reasons": {
                "no_vesicle_pixels": 0,
                "not_enough_pixels": 0,
                "dimensions_too_small": 0,
                "cannot_fit_box": 0,
                "other_errors": 0
            },
            "per_bbox": {}
        }
        
        # Check each bounding box
        for bbox_name, (raw_vol, seg_vol, add_mask_vol) in self.vol_data_dict.items():
            # Initialize per-bbox stats
            self.results["per_bbox"][bbox_name] = {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "skip_reasons": {
                    "no_vesicle_pixels": 0,
                    "not_enough_pixels": 0,
                    "dimensions_too_small": 0,
                    "cannot_fit_box": 0,
                    "other_errors": 0
                }
            }
            
            # Get samples for this bbox
            bbox_samples = self.synapse_df[self.synapse_df['bbox_name'] == bbox_name]
            self.results["per_bbox"][bbox_name]["total_samples"] = len(bbox_samples)
            self.results["total_samples"] += len(bbox_samples)
            
            # Process each sample in this bbox
            for idx, syn_info in tqdm(bbox_samples.iterrows(), 
                                      total=len(bbox_samples), 
                                      desc=f"Processing {bbox_name}"):
                log_capture_handler.reset()
                
                # Get coordinates
                central_coord = (int(syn_info['central_coord_1']), 
                                int(syn_info['central_coord_2']), 
                                int(syn_info['central_coord_3']))
                side1_coord = (int(syn_info['side_1_coord_1']), 
                              int(syn_info['side_1_coord_2']), 
                              int(syn_info['side_1_coord_3']))
                side2_coord = (int(syn_info['side_2_coord_1']), 
                              int(syn_info['side_2_coord_2']), 
                              int(syn_info['side_2_coord_3']))
                
                # Attempt to create segmented cube with type 12
                try:
                    overlaid_cube = self.data_loader.create_segmented_cube(
                        raw_vol=raw_vol,
                        seg_vol=seg_vol,
                        add_mask_vol=add_mask_vol,
                        central_coord=central_coord,
                        side1_coord=side1_coord,
                        side2_coord=side2_coord,
                        segmentation_type=12,  # Always use type 12
                        bbox_name=bbox_name,
                        vesicle_fill_threshold=vesicle_fill_threshold
                    )
                    
                    # If we got a result, count it as processed
                    if overlaid_cube is not None:
                        self.results["processed_samples"] += 1
                        self.results["per_bbox"][bbox_name]["processed_samples"] += 1
                    else:
                        # Determine skip reason from captured logs
                        self.results["skipped_samples"] += 1
                        self.results["per_bbox"][bbox_name]["skipped_samples"] += 1
                        
                        skip_reason = log_capture_handler.get_skip_reason()
                        if "No vesicle pixels" in skip_reason:
                            self.results["skip_reasons"]["no_vesicle_pixels"] += 1
                            self.results["per_bbox"][bbox_name]["skip_reasons"]["no_vesicle_pixels"] += 1
                        elif "too small" in skip_reason and "minimum required" in skip_reason:
                            self.results["skip_reasons"]["not_enough_pixels"] += 1
                            self.results["per_bbox"][bbox_name]["skip_reasons"]["not_enough_pixels"] += 1
                        elif "dimensions too small" in skip_reason:
                            self.results["skip_reasons"]["dimensions_too_small"] += 1
                            self.results["per_bbox"][bbox_name]["skip_reasons"]["dimensions_too_small"] += 1
                        elif "Cannot fit" in skip_reason:
                            self.results["skip_reasons"]["cannot_fit_box"] += 1
                            self.results["per_bbox"][bbox_name]["skip_reasons"]["cannot_fit_box"] += 1
                        else:
                            self.results["skip_reasons"]["other_errors"] += 1
                            self.results["per_bbox"][bbox_name]["skip_reasons"]["other_errors"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing sample {idx} in {bbox_name}: {e}")
                    self.results["skipped_samples"] += 1
                    self.results["skip_reasons"]["other_errors"] += 1
                    self.results["per_bbox"][bbox_name]["skipped_samples"] += 1
                    self.results["per_bbox"][bbox_name]["skip_reasons"]["other_errors"] += 1
        
        # Calculate percentages
        if self.results["total_samples"] > 0:
            self.results["processed_percentage"] = 100 * self.results["processed_samples"] / self.results["total_samples"]
            self.results["skipped_percentage"] = 100 * self.results["skipped_samples"] / self.results["total_samples"]
            
            for bbox_name in self.results["per_bbox"]:
                bbox_total = self.results["per_bbox"][bbox_name]["total_samples"]
                if bbox_total > 0:
                    self.results["per_bbox"][bbox_name]["processed_percentage"] = (
                        100 * self.results["per_bbox"][bbox_name]["processed_samples"] / bbox_total
                    )
                    self.results["per_bbox"][bbox_name]["skipped_percentage"] = (
                        100 * self.results["per_bbox"][bbox_name]["skipped_samples"] / bbox_total
                    )
        
        # Remove the custom handler
        logger.removeHandler(log_capture_handler)
        
        return self.results
    
    def print_report(self):
        """Print a formatted report of the sanity check results."""
        print("\n" + "="*80)
        print(" SEGMENTATION TYPE 12 SANITY CHECK REPORT ")
        print("="*80)
        
        print(f"\nTotal samples: {self.results['total_samples']}")
        print(f"Processed samples: {self.results['processed_samples']} ({self.results.get('processed_percentage', 0):.2f}%)")
        print(f"Skipped samples: {self.results['skipped_samples']} ({self.results.get('skipped_percentage', 0):.2f}%)")
        
        print("\nSkip reasons:")
        for reason, count in self.results["skip_reasons"].items():
            if self.results["skipped_samples"] > 0:
                percentage = 100 * count / self.results["skipped_samples"]
            else:
                percentage = 0
            print(f"  - {reason}: {count} ({percentage:.2f}% of skipped)")
        
        print("\nPer-bounding box results:")
        for bbox_name, stats in self.results["per_bbox"].items():
            print(f"\n{bbox_name}:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Processed: {stats['processed_samples']} ({stats.get('processed_percentage', 0):.2f}%)")
            print(f"  Skipped: {stats['skipped_samples']} ({stats.get('skipped_percentage', 0):.2f}%)")
            
            if stats["skipped_samples"] > 0:
                print("  Skip reasons:")
                for reason, count in stats["skip_reasons"].items():
                    if count > 0:
                        percentage = 100 * count / stats["skipped_samples"]
                        print(f"    - {reason}: {count} ({percentage:.2f}% of skipped)")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path="segtype12_sanity_check_results.csv"):
        """
        Save the detailed results to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        # Create a list of records for the DataFrame
        records = []
        
        # Add overall stats
        records.append({
            "bbox_name": "ALL",
            "total_samples": self.results["total_samples"],
            "processed_samples": self.results["processed_samples"],
            "skipped_samples": self.results["skipped_samples"],
            "processed_percentage": self.results.get("processed_percentage", 0),
            "skipped_percentage": self.results.get("skipped_percentage", 0),
            **{f"skip_{reason}": count for reason, count in self.results["skip_reasons"].items()}
        })
        
        # Add per-bbox stats
        for bbox_name, stats in self.results["per_bbox"].items():
            records.append({
                "bbox_name": bbox_name,
                "total_samples": stats["total_samples"],
                "processed_samples": stats["processed_samples"],
                "skipped_samples": stats["skipped_samples"],
                "processed_percentage": stats.get("processed_percentage", 0),
                "skipped_percentage": stats.get("skipped_percentage", 0),
                **{f"skip_{reason}": count for reason, count in stats["skip_reasons"].items()}
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed report to {output_path}")


class SkipReasonHandler(logging.Handler):
    """Custom logging handler to capture skip reason messages."""
    
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def emit(self, record):
        msg = self.format(record)
        if "Discarding sample" in msg:
            self.messages.append(msg)
    
    def get_skip_reason(self):
        """Return the captured skip reason message."""
        if self.messages:
            return self.messages[0]
        return ""
    
    def reset(self):
        """Clear captured messages."""
        self.messages = []


def main():
    """Run the sanity check with default parameters."""
    # Create and run the sanity check
    checker = SegType12SanityCheck()
    
    # Check with 100% fill requirement
    results_100 = checker.run_check(vesicle_fill_threshold=100.0)
    checker.print_report()
    checker.save_report("segtype12_sanity_check_100pct.csv")
    
    # Also check with 95% fill requirement for comparison
    results_95 = checker.run_check(vesicle_fill_threshold=95.0)
    checker.print_report()
    checker.save_report("segtype12_sanity_check_95pct.csv")
    
    # And with 80% fill requirement
    results_80 = checker.run_check(vesicle_fill_threshold=80.0)
    checker.print_report()
    checker.save_report("segtype12_sanity_check_80pct.csv")
    
    # Print comparison
    print("\n" + "="*80)
    print(" COMPARISON OF DIFFERENT FILL THRESHOLDS ")
    print("="*80)
    print(f"100% fill: {results_100['processed_samples']} processed, {results_100['skipped_samples']} skipped ({results_100.get('processed_percentage', 0):.2f}% success)")
    print(f"95% fill: {results_95['processed_samples']} processed, {results_95['skipped_samples']} skipped ({results_95.get('processed_percentage', 0):.2f}% success)")
    print(f"80% fill: {results_80['processed_samples']} processed, {results_80['skipped_samples']} skipped ({results_80.get('processed_percentage', 0):.2f}% success)")


if __name__ == "__main__":
    main() 