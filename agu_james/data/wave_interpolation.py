# Derived from npz/resize_swh.py
# See provenance/sources.json for original hash and modifications.
"""
Resize 'swh' variable in sfc_regular npz files from 96x96 to 192x192 using bilinear interpolation.
"""
from agu_james import runtime
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
import glob
from tqdm import tqdm
import shutil

def resize_swh_bilinear(swh_data, target_size=(192, 192)):
    """
    Resize swh data from current size to target size using bilinear interpolation.
    
    Args:
        swh_data (numpy.ndarray): Input swh data with shape (96, 96)
        target_size (tuple): Target size (192, 192)
    
    Returns:
        numpy.ndarray: Resized swh data with shape (192, 192)
    """
    current_shape = swh_data.shape
    zoom_factors = [target_size[i] / current_shape[i] for i in range(len(current_shape))]
    resized_data = zoom(swh_data, zoom_factors, order=1, mode='reflect')
    return resized_data
