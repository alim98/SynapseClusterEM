
"""
CONSISTENT GRAY VALUE GUIDE

This guide explains how to maintain consistent gray values across the entire pipeline
from dataset loading to GIF creation.

KEY SETTINGS TO MAINTAIN CONSISTENT GRAY VALUES:

1. During Dataset Initialization:
   - Set processor.normalize_volume = False
   - Set normalize_across_volume = False
   Example:
     ```
     processor = Synapse3DProcessor(size=config.size)
     processor.normalize_volume = False
     
     dataset = SynapseDataset(
         vol_data_dict=vol_data_dict,
         synapse_df=syn_df,
         processor=processor,
         segmentation_type=config.segmentation_type,
         subvol_size=config.subvol_size,
         num_frames=config.num_frames,
         alpha=config.alpha,
         normalize_across_volume=False  # This is crucial!
     )
     ```

2. During GIF Creation:
   - Use create_gif_from_volume WITHOUT normalization
   - The function in GifUmap.py has been modified to NEVER apply normalization
   Example:
     ```
     # The create_gif_from_volume function has been modified to:
     # - Never apply normalization
     # - Preserve the original consistent gray values
     gif_path, frames = create_gif_from_volume(volume, str(output_path), fps=5)
     ```

3. How to test if gray values are consistent:
   Run the test_gif_creation.py script, which will:
   - Initialize the dataset with normalization disabled
   - Create GIFs that preserve the original gray values
   - Output GIFs to the test_output_real directory

TESTING:
To test whether the gray values are consistent:
1. Create GIFs using the code examples above
2. Visually inspect the GIFs:
   - Consistent gray values: The background intensity should look similar across all frames
   - Inconsistent gray values: The background may appear brighter in some frames, darker in others

If your GIFs have inconsistent gray values, check:
1. That you're using the modified create_gif_from_volume function without normalization
2. That the dataset is initialized with normalization disabled
3. That the SynapseDataLoader is configured correctly (check initialize_dataset_from_newdl function)

CODE LOCATIONS THAT HAVE BEEN FIXED:

1. In synapse/gif_umap/GifUmap.py:
   - The create_gif_from_volume function has been modified to never apply normalization
   - The initialize_dataset_from_newdl function sets processor.normalize_volume = False and 
     normalize_across_volume = False

2. In test_gif_creation.py:
   - The create_modified_gif_from_volume function allows controlling normalization
   - You can use apply_normalization=False to preserve consistent gray values

For consistent visualization, ensure all places that output images (including slice views)
use consistent normalization approaches. For debugging, you can generate both normalized
and non-normalized versions to compare the difference.
""" 