def __getitem__(self, idx):
    """
    Get a preprocessed synapse cube, its metadata and bounding box name.
    
    Args:
        idx: Index of the synapse in the dataframe
        
    Returns:
        tuple: (processed_cube, synapse_info, bbox_name)
    """
    synapse_info = self.synapse_df.iloc[idx]
    bbox_name = synapse_info['bbox_name']
    
    if bbox_name in self.vol_data_dict:
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict[bbox_name]
        
        # Get coordinates
        central_coord = (
            int(synapse_info['central_coord_1']),
            int(synapse_info['central_coord_2']),
            int(synapse_info['central_coord_3'])
        )
        side1_coord = (
            int(synapse_info['side_1_coord_1']),
            int(synapse_info['side_1_coord_2']),
            int(synapse_info['side_1_coord_3'])
        )
        side2_coord = (
            int(synapse_info['side_2_coord_1']),
            int(synapse_info['side_2_coord_2']),
            int(synapse_info['side_2_coord_3'])
        )
        
        # Check if intelligent cropping is enabled
        use_intelligent_cropping = getattr(self.processor, 'use_intelligent_cropping', False)
        presynapse_weight = getattr(self.processor, 'presynapse_weight', 0.7)
        normalize_presynapse_size = getattr(self.processor, 'normalize_presynapse_size', False)
        
        # Get data loader
        data_loader = self.processor.data_loader
        
        # Create cube with intelligent cropping settings if enabled
        cube = data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.processor.size[0],
            alpha=self.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=True,
            smart_crop=use_intelligent_cropping,  # Use intelligent cropping if enabled
            presynapse_weight=presynapse_weight,  # Weight for presynapse shift
            normalize_presynapse_size=normalize_presynapse_size  # Normalize presynapse size if enabled
        )
        
        # Process cube
        processed_cube = self.processor.process_cube(cube)
        return processed_cube, synapse_info, bbox_name
    else:
        raise ValueError(f"Bounding box {bbox_name} not found in volume data dictionary") 