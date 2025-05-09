import numpy as np

# ==================WEBKNOSSOS UTILS===================

import webknossos as wk
from webknossos import BoundingBox

def get_dataset_layer_mag(ds_path, layer_name, mag):
    ds = wk.Dataset.open(ds_path)
    layer = ds.get_layer(layer_name)
    mag = layer.get_mag(mag)
    return mag

def read_mag_bbox_data(mag, center, size):
    center = np.array(center)
    size = np.array(size)
    bb = BoundingBox(
        topleft = center - size // 2,
        size = size
    )
    return mag.read(absolute_bounding_box=bb)

# ==================MASK UTILS========================

from skimage.measure import label, regionprops

def filter_non_center_components(mask):
    """
    Filter out the non-center components of a mask. 
    The center component is the one closest to the center of the mask.
    """

    mask = label(mask)
    center = np.array(mask.shape) // 2
    props = regionprops(mask)

    min_dist_idx = 0
    min_dist = np.inf
    for i,prop in enumerate(props):
        dists = np.linalg.norm(prop.coords - center, axis=1)
        dist = np.min(dists)
        if dist < min_dist:
            min_dist = dist
            min_dist_idx = i

    new_mask = np.zeros_like(mask)
    idxs = props[min_dist_idx].coords
    new_mask[tuple(idxs.T)] = 1
    
    return new_mask


#==================BBOX UTILS========================

def get_point_mask_in_boxes(points, boxes):
    """
    Create a mask for points inside given 3D boxes.
    """
    points = np.array(points)
    boxes = np.array(boxes)
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes:
        min_x, min_y, min_z, w, h, d = box
        max_x = min_x + w
        max_y = min_y + h
        max_z = min_z + d
        inside = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
            (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
        )
        mask |= inside  # Accumulate any point inside any box
    return mask