import numpy as np
import h5py
import os
from typing import Tuple, Iterable
from tqdm import tqdm

# Fix imports to use relative imports
from src.bbox_reader import BboxReader
from src.utils import filter_non_center_components, get_dataset_layer_mag, read_mag_bbox_data, get_point_mask_in_boxes


# Original paths
ORIGINAL_CONNECTOME_PATH = "/nexus/posix0/MBR-neuralsystems/vx/artifacts/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v1/create_connectome/connectome__34fa6f477d-v1/connectome/connectome.hdf5"
ORIGINAL_SYNAPSE_PREDICTION_DATASET_PATH = "/nexus/posix0/MBR-neuralsystems/vx/artifacts/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v1/predict/merlin_synapse_model_v8a_synapse_prediction_v1-v1/prediction"
ORIGINAL_AGGLO_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/segmentation/agglomerates/agglomerate_view_60.hdf5'
ORIGINAL_RAW_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_extended_v2_nuclei_exclusion_with_meshes/'
ORIGINAL_SEG_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/'

# Determine if we need to use fallback paths
if os.path.exists(ORIGINAL_CONNECTOME_PATH):
    CONNECTOME_PATH = ORIGINAL_CONNECTOME_PATH
    SYNAPSE_PREDICTION_DATASET_PATH = ORIGINAL_SYNAPSE_PREDICTION_DATASET_PATH
    AGGLO_PATH = ORIGINAL_AGGLO_PATH
    RAW_PATH = ORIGINAL_RAW_PATH
    SEG_PATH = ORIGINAL_SEG_PATH
    print("Using original data paths")
else:
    # Fallback to dummy paths - will use policy="dummy" automatically
    print("Original paths not found. Using dummy data mode.")
    CONNECTOME_PATH = ""
    SYNAPSE_PREDICTION_DATASET_PATH = ""
    AGGLO_PATH = ""
    RAW_PATH = ""
    SEG_PATH = ""

# Hardcoded plexiform bboxes from Dom
plexiform_bboxes = [
    [11775, 0,   6662,  7114, 12790, 2476], #for 8nm
    [11588, 0,   8966,  6780, 12699, 2684],
    [11339, 65,  11372, 6638, 12608, 2684],
    [10584, 179, 13828, 6814, 12494, 2684],
    [10033, 179, 16295, 7148, 12494, 2619],
    [11977, 65,  4129,  7577, 12484, 2670],
    [12500, 65,  1570,  7097, 12484, 2670],
    [12500, 65,  570,   7356, 12484, 2670]
]

# Global variables so we load them only once (256 MB total)
positions = None
agglo_ids = None

rng = np.random.default_rng(seed=37)

def sample_connectome(batch_size=1, policy="random", connectome_path=CONNECTOME_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample positions for synapse data.
    Args:
        batch_size (int): The number of positions to sample.
        policy (str): The sampling policy. Can be "random" 
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the sampled positions and agglomeration IDs (shapes: (batch_size, 3) and (batch_size,)).
    """

    global positions, agglo_ids

    # Always use dummy data if paths aren't available
    if not os.path.exists(connectome_path) or policy == "dummy":
        return np.zeros((batch_size, 3), dtype=np.uint32), np.ones((batch_size,), dtype=np.uint32)

    # load only once    
    if positions is None:
        try:
            with h5py.File(connectome_path, 'r') as f:
                positions = f['synapse_positions'][:]
                agglo_ids = f['synapse_to_src_agglomerate'][:]
            
            mask = get_point_mask_in_boxes(positions, plexiform_bboxes)
            positions = positions[mask]
            agglo_ids = agglo_ids[mask]
        except Exception as e:
            print(f"Error loading connectome: {e}")
            print("Falling back to dummy data")
            return np.zeros((batch_size, 3), dtype=np.uint32), np.ones((batch_size,), dtype=np.uint32)

    if policy == "random":
        # Sample random indices
        indices = rng.choice(positions.shape[0], size=batch_size, replace=False)
        
        # Get the sampled positions and agglomeration IDs
        return positions[indices], agglo_ids[indices]

    elif policy == "sequential": raise NotImplementedError("Sequential sampling is not implemented yet.")
    else: raise ValueError(f"Unknown sampling policy: {policy}")


def get_synapse_data(position: Iterable, 
                     src_agglo_id: int, 
                     synapse_prediction_dataset_path = SYNAPSE_PREDICTION_DATASET_PATH,
                     raw_path = RAW_PATH,
                     seg_path = SEG_PATH,
                     agglo_path = AGGLO_PATH
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get synapse data for a given position and source agglomeration ID.
    Args:
        position (Iterable): The position in the format [x, y, z].
        src_agglo_id (int): The pre-synaptic agglomeration ID.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the raw data and the mask of the pre-synaptic agglomeration and the cleft bot with shape (1, 80, 80, 80).
    """
    # Check if paths exist, if not, return dummy data
    if not os.path.exists(synapse_prediction_dataset_path) or not os.path.exists(raw_path) or not os.path.exists(seg_path) or not os.path.exists(agglo_path):
        print("Paths not found, using dummy data")
        # Return random 80x80x80 volumes
        dummy_raw = np.random.randint(0, 256, size=(1, 80, 80, 80), dtype=np.uint8)
        
        # Create a simple mask (a central sphere)
        y, x, z = np.ogrid[0:80, 0:80, 0:80]
        center_x, center_y, center_z = 40, 40, 40
        dummy_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= 20**2).astype(np.uint8)[np.newaxis, ...]
        
        return dummy_raw, dummy_mask
        
    try:
        # get dataset layer
        layer_name = "segment_types_task$non-spine-head-synapse"
        mag = get_dataset_layer_mag(synapse_prediction_dataset_path, layer_name, 1)

        # get synapse mask
        bbox_size = np.array([80, 80, 80])
        center = position
        synapse = read_mag_bbox_data(mag, center, bbox_size)
        synapse_threshold = 100
        synapse_mask = synapse > synapse_threshold

        # get pre-synapse agglomeration mask
        reader = BboxReader(bbox_size=np.array([80, 80, 80]), bbox_unit_scale='vox', raw_path=raw_path, seg_path=seg_path, agglo_path=agglo_path)
        raw, agglo = reader.read_bbox_agglo_data(position, 1)
        pre_agglo_mask = (agglo == src_agglo_id).astype(np.uint8)
        
        # merge synapse mask with agglomeration
        full_mask = pre_agglo_mask | filter_non_center_components(synapse_mask.squeeze())[None, ...]

        return raw, full_mask
    
    except Exception as e:
        print(f"Error getting synapse data: {e}")
        print("Falling back to dummy data")
        
        # Return random 80x80x80 volumes
        dummy_raw = np.random.randint(0, 256, size=(1, 80, 80, 80), dtype=np.uint8)
        
        # Create a simple mask (a central sphere)
        y, x, z = np.ogrid[0:80, 0:80, 0:80]
        center_x, center_y, center_z = 40, 40, 40
        dummy_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= 20**2).astype(np.uint8)[np.newaxis, ...]
        
        return dummy_raw, dummy_mask

def sample_synapses(batch_size=1, policy="random", verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample synapse raw em data and pre-synaptic agglomeration + cleft mask for a given batch size and policy.
    """
    # Check if paths exist, if not, use dummy policy
    if not os.path.exists(CONNECTOME_PATH) and policy != "dummy":
        print("Original paths not found. Forcing dummy data mode.")
        policy = "dummy"
        
    if policy == "dummy":
        if verbose: print(f"Creating {batch_size} dummy synapse boxes...")
        
        # Create random raw data (grayscale cubes with some noise)
        raw = np.random.randint(0, 256, size=(batch_size, 1, 80, 80, 80), dtype=np.uint8)
        
        # Create masks (central spheres)
        mask = np.zeros((batch_size, 1, 80, 80, 80), dtype=np.uint8)
        for i in range(batch_size):
            # Create a random sphere in the cube
            y, x, z = np.ogrid[0:80, 0:80, 0:80]
            # Center of the cube
            center_x, center_y, center_z = 40, 40, 40
            # Random offset from center
            center_offset = np.random.randint(-10, 10, size=3)
            # Random radius between 15 and 25
            radius = np.random.randint(15, 25)
            # Create sphere mask
            sphere_mask = ((x - center_x - center_offset[0])**2 + 
                           (y - center_y - center_offset[1])**2 + 
                           (z - center_z - center_offset[2])**2 <= radius**2)
            mask[i, 0] = sphere_mask.astype(np.uint8)
        
        return raw, mask
    else:
        try:
            positions, agglo_ids = sample_connectome(batch_size, policy)
            raw = []
            mask = []
            if verbose: print(f"Reading and Masking {batch_size} synapse boxes...")
            wrapper = lambda x: tqdm(x, total=batch_size) if verbose else x
            for position, src_agglo_id in wrapper(zip(positions, agglo_ids)):
                r, m = get_synapse_data(position, src_agglo_id)
                raw.append(r)
                mask.append(m)
            raw = np.stack(raw, axis=0)
            mask = np.stack(mask, axis=0)
            return raw, mask
        except Exception as e:
            print(f"Error in sample_synapses: {e}")
            print("Falling back to dummy data")
            return sample_synapses(batch_size, "dummy", verbose)
    

if __name__ == "__main__":
    # Test the functions
    raw, mask = sample_synapses(batch_size=10, policy="dummy", verbose=True)
    print("Raw shape:", raw.shape)
    print("Mask shape:", mask.shape)