# Derived from npz/fill_nan_nearest.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

def nearest_neighbor_fill(data):
    """
    使用最近鄰方法填補NaN值
    
    Parameters:
    data: 2D numpy array with potential NaN values
    
    Returns:
    filled_data: 2D numpy array with NaN values filled
    """
    if not np.isnan(data).any():
        return data
    filled_data = data.copy()
    rows, cols = data.shape
    row_coords, col_coords = np.mgrid[0:rows, 0:cols]
    coords = np.column_stack([row_coords.ravel(), col_coords.ravel()])
    valid_mask = ~np.isnan(data.ravel())
    valid_coords = coords[valid_mask]
    valid_values = data.ravel()[valid_mask]
    nan_mask = np.isnan(data.ravel())
    nan_coords = coords[nan_mask]
    if len(valid_coords) == 0:
        print('警告：所有值都是NaN，無法進行填補')
        return filled_data
    if len(nan_coords) == 0:
        return filled_data
    tree = cKDTree(valid_coords)
    distances, indices = tree.query(nan_coords)
    filled_values = valid_values[indices]
    filled_data_flat = filled_data.ravel()
    filled_data_flat[nan_mask] = filled_values
    filled_data = filled_data_flat.reshape(rows, cols)
    return filled_data
