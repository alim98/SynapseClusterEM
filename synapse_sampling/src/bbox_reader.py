import numpy as np
import h5py
import webknossos as wk

class BboxReader():

    def __init__(self,
        # Paths
        agglo_path = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/segmentation/agglomerates/agglomerate_view_60.hdf5',
        seg_path = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/',
        raw_path = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_extended_v2_nuclei_exclusion_with_meshes/',

        # Constants
        voxel_size = [8, 8, 8],  # nm (dataset resolution)
        bbox_size = [4, 4, 4],  # µm
        bbox_unit_scale: str = "µm", # scale to nm (ex: 1e3 for µm) or None for 1vox
        spacing = 10,  # µm (for nml skeleton sampling)
        min_distance = 20,  # µm (for nml skeleton sampling)
    ) -> None:

        self.AGGLO_PATH = agglo_path
        self.SEG_PATH = seg_path
        self.RAW_PATH = raw_path

        self.VOXEL_SIZE = np.array(voxel_size)
        self.SPACING = spacing
        self.BBOX_SIZE = np.array(bbox_size)
        #Replaced match with if-statements(for version compatibility)
        if bbox_unit_scale in ["µm", "um"]:  # Handle both "µm" and "um"
            self.BBOX_UNIT_SCALE = 1e3  # Convert µm to nm
        elif bbox_unit_scale == "nm":
            self.BBOX_UNIT_SCALE = 1    # Already in nm
        elif bbox_unit_scale == "vox":
            self.BBOX_UNIT_SCALE = self.VOXEL_SIZE[0]    # Default to voxel units
        else:
            raise ValueError(f"Unknown unit scale: {bbox_unit_scale}. Use 'µm', 'nm', or 'vox'.")

        self.update_vox_size()

    def update_vox_size(self):
        self.SPACING_VOX = int((self.SPACING * self.BBOX_UNIT_SCALE) / self.VOXEL_SIZE[0]) # minimum spacing between nodes in voxels
        self.BBOX_SIZE_VOX = (self.BBOX_SIZE * self.BBOX_UNIT_SCALE / self.VOXEL_SIZE).astype(int)
        

    def get_seg_from_agglo(self, agglo_ids):
        """Get segment ids for given agglomerate IDs."""
        with h5py.File(self.AGGLO_PATH, "r") as f:
            # Get the agglos_to_seg mapping and its offset
            agglos_to_seg = f["agglomerate_to_segments"][:] 
            agglos_offset = f["agglomerate_to_segments_offsets"][:] 
            segment_ids_for_agglos = [agglos_to_seg[agglos_offset[agglo_id] : agglos_offset[agglo_id+1]] 
                                    for agglo_id in agglo_ids]
        return segment_ids_for_agglos

    def get_agglo_from_seg(self, seg_ids):
        """Get agglomerate IDs for given segment IDs."""
        with h5py.File(self.AGGLO_PATH, "r") as f:
            # Get the seg_to_agglos mapping and its offset
            seg_to_agglos = f["segment_to_agglomerate"][:] 
            agglos_for_segments = seg_to_agglos[seg_ids]
        return agglos_for_segments

    def get_bbox_topleft(self, point):
        """Get the top-left corner of the bounding box centered on point."""
        half_size = self.BBOX_SIZE_VOX // 2
        start = point - half_size
        # align to magnification
        start = np.round(start / self.VOXEL_SIZE) * self.VOXEL_SIZE
        return start
    
    def get_bbox_size(self):
        """Get the size of the bounding box in voxels."""
        return self.BBOX_SIZE_VOX.astype(int)

    def read_bbox_data(self, point, magnification):
        """Read raw and segmentation data for a bbox centered on point."""

        raw_data = self.read_bbox(point, self.SEG_PATH, "segmentation", magnification)
        seg_data = self.read_bbox(point, self.RAW_PATH, "color_InLens", magnification)
        
        # Convert raw_data from uint16 to uint8 and invert
        raw_data = raw_data.astype(np.float32)  # Convert to float first to avoid overflow
        raw_data = (raw_data / 256).astype(np.uint8)  # Scale down to uint8 range
        raw_data = 255 - raw_data  # Invert the values
        
        return raw_data, seg_data

    def read_bbox_agglo_data(self, point, magnification):
        raw_data, seg_data = self.read_bbox_data(point, magnification)
        agglo_data = self.get_agglo_from_seg(seg_data)  
        return raw_data, agglo_data

    def read_bbox(self, point, ds_path, layer_name, magnififcation):
        ds = wk.Dataset.open(ds_path)
        layer = ds.get_layer(layer_name)
        mag = layer.get_mag(magnififcation)
        start = self.get_bbox_topleft(point)
        if np.any(start < 0): print("Bbox start position is out of bounds.")
        return mag.read(absolute_offset=start.astype(int), size=self.BBOX_SIZE_VOX.astype(int))
    











