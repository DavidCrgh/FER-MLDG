"""
util.py

This module provides utility functions for other modules. It defines constants
for dataframe column names and functions to get image and annotation directories.

Constants:
    DF_COLS

Functions:
    get_img_dir
    get_anno_dir
"""
__all__ = ['DF_COLS', 'get_img_dir', 'get_anno_dir']

# Define main dataframe column names
DF_COLS = {
    'file_col' : 'file', 
    'filename_col' : 'filename', 
    'anno_col' : 'label', 
    'set_col' : 'set'}


def get_img_dir(path : str, subset : str) -> str:
    """
    Returns the image directory path for the given subset.
    
    Args:
        path (str): The dataset directory path.
        subset (str): The subset (e.g., 'train', 'val').



    Returns:
        str: The image directory path.
    """
    
    return path + subset + "_set" + "/images/"


def get_anno_dir(path : str, subset : str) -> str:
    """
    Constructs the directory path for annotations based on the given base path and subset.
    Args:
        path (str): The base directory path.
        subset (str): The subset name to be appended to the path.
    Returns:
        str: The constructed directory path for annotations.
    """

    return path + subset + "_set" + "/annotations/"