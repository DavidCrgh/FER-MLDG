"""
align.py

This module provides the functionality to align the dataset face images using
RetinaFace as a backend. It includes functions to add margins to images, align
face images, and process the entire dataset.

Constants:
    PAD_DEF
    HEIGHT_DEF
    WIDTH_DEF

Functions:
    process_dataset
"""
__all__ = ['PAD_DEF', 'HEIGHT_DEF', 'WIDTH_DEF', 'process_dataset']

import os

from retinaface import RetinaFace
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as T_Image

from tqdm import tqdm

import warnings
from typing import Callable

from argparse import Namespace

from lib.util import DF_COLS, get_img_dir

PAD_DEF = 0.5

HEIGHT_DEF = 224
WIDTH_DEF = 224

aligned_faces = 0

def add_margin(
        pil_img : Image, 
        top : int, 
        right : int, 
        bottom : int, 
        left : int, 
        color : tuple[int, int, int] = (255, 255, 255)) -> Image:
    """
    Adds a margin to the given PIL image.

    Args:
        pil_img (Image): The original PIL image.
        top (int): The size of the top margin.
        right (int): The size of the right margin.
        bottom (int): The size of the bottom margin.
        left (int): The size of the left margin.
        color (tuple[int, int, int], optional): The color of the margin. Defaults to white (255, 255, 255).

    Returns:
        Image: The new image with the added margins.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result

def align_face_img(
        img_path : str, 
        height : int = HEIGHT_DEF, 
        width : int = WIDTH_DEF,
        pad_rate : float = PAD_DEF) -> Image:
    """
    Aligns the face in the image located at the given path.

    Args:
        img_path (str): The path to the image file.
        height (int, optional): The height of the output image. Defaults to HEIGHT_DEF.
        width (int, optional): The width of the output image. Defaults to WIDTH_DEF.
        pad_rate (float, optional): The padding rate to apply to the image. Defaults to PAD_DEF.

    Returns:
        Image: The aligned and resized image.
    """
    global aligned_faces

    # Load image from disk and add white margins to improve RF's detection
    img = Image.open(img_path)
    
    if pad_rate > 0.0:
        h_margin = int((img.size[0] * pad_rate) // 2)
        v_margin = int((img.size[1] * pad_rate) // 2)

        img = add_margin(img, top=v_margin, bottom=v_margin, left=h_margin, right=h_margin)

    # Detect and aligns all faces in img
    img_np = np.asarray(img)[:,:,::-1] # Convert RGB -> BGR
    faces = RetinaFace.extract_faces(img_np, align=True)

    if len(faces) < 1:
        warnings.warn(f'Unable to detect face in image {img_path}. Original image will be used.')
        img = Image.open(img_path)

    else:
        img = Image.fromarray(faces[0])
        aligned_faces += 1

        if len(faces) > 1:
            warnings.warn(f'Detected more than one face in image {img_path}. First detected face will be used.')


    img_resized = img.resize((width, height))

    return img_resized

def process_dataset(
        data : pd.DataFrame, 
        args : Namespace,
        save_func : Callable[[T_Image, pd.Series], None]) -> None:
    """
    Processes the dataset by aligning faces in images and saving the results.

    Args:
        data (pd.DataFrame): The dataframe containing the dataset information.
        args (Namespace): The arguments containing the configuration for processing.
        save_func (Callable[[T_Image, pd.Series], None]): The function to save the processed image.
    """
    global aligned_faces
    aligned_faces = 0

    total = len(data)
    tqdm_iter = tqdm(data.iterrows(), total=total, desc="Processing images", mininterval=30.0)

    for idx, row in tqdm_iter:
        file = row[DF_COLS['file_col']]
        _set = row[DF_COLS['set_col']]

        #tqdm_iter.set_postfix(file=f"{_set}_set/images/{file}")

        img_dir = get_img_dir(path=args.path, subset=_set)
        img_path = os.path.join(img_dir, file)

        img = align_face_img(
            img_path, 
            height=args.out_height,
            width=args.out_width,
            pad_rate=args.padding)

        if save_func:
            save_func(img, row)

    aligned_percent = aligned_faces / len(data)
    print(f'\nAligned {aligned_faces} faces out of {len(data)} images ({(aligned_percent * 100):.2f}%).')