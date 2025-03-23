def calculate_bbox_coordinates(central_coord_x, central_coord_y, central_coord_z, bb_num):
    """
    Calculate bbox coordinates based on central coordinates.
    
    Args:
        central_coord_x (int): Central x coordinate
        central_coord_y (int): Central y coordinate 
        central_coord_z (int): Central z coordinate
        
    Returns:
        tuple: (x1, y1, z1) coordinates for bbox
    """
    # Get base coordinates based on bounding box number
    if bb_num == 6:
        x1 = 16700 - 100 + central_coord_x
        y1 = 11200 - 100 + central_coord_y
        z1 = 15573 - 100 + central_coord_z
    elif bb_num == 2:
        x1 = 18544  + central_coord_x
        y1 = 5409 + central_coord_y
        z1 = 4374 + central_coord_z
    elif bb_num == 5:
        x1 = 1782 - 100 + central_coord_x
        y1 = 8962 - 100 + central_coord_y
        z1 = 8309 - 100 + central_coord_z
    elif bb_num == 4:
        x1 = 15813 - 100 + central_coord_x
        y1 = 7009 - 100 + central_coord_y
        z1 = 10408 - 100 + central_coord_z
    elif bb_num == 1:
        x1 = 14219 - 100 + central_coord_x
        y1 = 10792 - 100 + central_coord_y
        z1 = 15134 - 100 + central_coord_z
    elif bb_num == 3:
        x1 = 14783 - 100 + central_coord_x
        y1 = 3707 - 100 + central_coord_y
        z1 = 4316 - 100 + central_coord_z
    elif bb_num == 7:
        x1 = 12340 + central_coord_x
        y1 = 9416 + central_coord_y
        z1 = 16439 + central_coord_z

    webknossos_url = f"https://webknossos.brain.mpg.de/annotations/67bcfa0301000006202da79c#{x1},{y1},{z1},0,0.905,1506"
    
    print(f"Bbox coordinates: x1={x1}, y1={y1}, z1={z1}")
    print(f"Webknossos URL: {webknossos_url}")
    
    return x1, y1, z1


# calculate_bbox_coordinates(402,378,441,7)
# # # calculate_bbox_coordinates(357,330,268,6)
# # calculate_bbox_coordinates(334,127,210,5)
# calculate_bbox_coordinates(442,202,345,2)