"""
data_io.py

This module implements utilities to read and write the data used by the preproc.py
script. It defines handlers for different output structures (APViT and EF) and
provides functions to read dataset headers and format output filenames and annotation
rows.

Classes:
    StructHandler
    APViTHandler
    EFHandler

Functions:
    read_dataset_heads
    format_out_filename
    format_anno_row
"""

__all__ = ['read_dataset_heads', 'StructHandler', 'APViTHandler', 'EFHandler']

import os
import shutil
import warnings

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from PIL.Image import Image

from lib.util import *
from tqdm import tqdm


# For replacing the occurrences of subset strings in output directories and
# files
SET_ALIASES = {'train' : 'train', 'val' : 'test'}


def format_out_filename(filename : str, _set : str) -> str:
    """
    Formats the output filename based on the subset.

    Args:
        filename (str): The original filename.
        _set (str): The subset (e.g., 'train', 'val').

    Returns:
        str: The formatted output filename.
    """
    return f'{SET_ALIASES[_set]}_{filename}_aligned.jpg'

def format_anno_row(row: pd.Series, _set : str) -> list:
    """
    Formats a row of annotations for output.

    Args:
        row (pd.Series): The row containing the annotation data.
        _set (str): The subset (e.g., 'train', 'val').

    Returns:
        list: A list containing the formatted filename and label.
    """
    filename = row[DF_COLS['filename_col']]
    label = row[DF_COLS['anno_col']]

    f_filename = format_out_filename(filename, _set)
    return [f_filename, label]

def read_dataset_heads(path : str, subset : str, append_set : bool = True) -> pd.DataFrame:
    """
    Reads the dataset headers and returns a dataframe.

    Args:
        path (str): The path to the dataset directory.
        subset (str): The subset to read (e.g., 'train', 'val').
        append_set (bool, optional): Whether to append the subset to the dataframe. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe containing the dataset headers.
    """
    img_dir = get_img_dir(path, subset)
    anno_dir = get_anno_dir(path, subset)

    data = [] # List of dicts

    listdir = os.listdir(img_dir)
    listdir = sorted(listdir, key= lambda x : int(x.split(sep='.')[0]))

    print(f'Found {len(listdir)} file(s) in {img_dir}')

    progress_bar = tqdm(listdir, desc="Reading dataset headers", total=len(listdir), mininterval=30.0)

    for file in progress_bar:
        filename = file
        filename_no_ext = filename.split(sep='.')[0]

        anno_filename = f'{filename_no_ext}_exp.npy'
        annotation = np.load(anno_dir + anno_filename)

        row = {DF_COLS['file_col']: filename,
               DF_COLS['filename_col']: filename_no_ext,
               DF_COLS['anno_col']: annotation.item()}

        if append_set:
            row[DF_COLS['set_col']] = subset

        data.append(row)

    return pd.DataFrame(data)


class StructHandler(ABC):
    """
    Abstract base class for handling different output structures.

    Args:
        data (pd.DataFrame): The dataframe containing the dataset information.
        struct_name (str): The name of the output structure.
        in_dir (str): The input directory.
        out_dir (str): The output directory.
    """
    def __init__(self, data : pd.DataFrame, struct_name : str, in_dir : str, out_dir : str) -> None:
        self.struct_name = struct_name

        self.train_count = 0
        self.val_count = 0

        self.in_dir = in_dir
        self.out_dir = os.path.join(out_dir, 'AffectNet/')
        
        self.data = data
        self.labels = data[DF_COLS['anno_col']].unique().tolist()
        self.sets = data[DF_COLS['set_col']].unique().tolist()

    @abstractmethod
    def init_dirs(self) -> None:
        """
        Initializes the output directory structure.
        """
        # Some shared code to initialize the base output directory
        print('\nInitializing output directory structure...')

        if os.path.exists(self.out_dir):
            warnings.warn(
                'Found "AffectNet" directory within output directory.' + 
                'Contents will be deleted!\n'
            )
            shutil.rmtree(self.out_dir)

        os.makedirs(self.out_dir)

        abs_path = os.path.abspath(self.out_dir)
        print(f'Preprocessed dataset contents will be written to {abs_path}\n')

    @abstractmethod
    def gen_labels(self) -> None:
        """
        Generates label files.
        """
        print('Generating label files...')

    @abstractmethod
    def save_img(self, img : Image, row : pd.Series) -> None:
        """
        Saves the processed image.

        Args:
            img (Image): The processed image.
            row (pd.Series): The row containing the image data.
        """
        pass

    def create_struct(self) -> None:
        """
        Creates the output structure by initializing directories and generating labels.
        """
        self.init_dirs()
        self.gen_labels()


class APViTHandler(StructHandler):
    """
    Handler for the APViT output structure.

    Args:
        data (pd.DataFrame): The dataframe containing the dataset information.
        in_dir (str): The input directory.
        out_dir (str): The output directory.
    """
    def __init__(self,data : pd.DataFrame, in_dir : str, out_dir : str) -> None:
        super().__init__(data=data, struct_name='APVIT', in_dir=in_dir, out_dir=out_dir)

        self.img_dir = None
        self.anno_dir = None

    def init_dirs(self) -> None:
        """
        Initializes the output directory structure for APViT.
        """
        super().init_dirs()
        
        self.img_dir = os.path.join(self.out_dir, 'basic/Image/aligned_224/')
        self.anno_dir = os.path.join(self.out_dir, 'basic/EmoLabel/')

        os.makedirs(self.img_dir)
        os.makedirs(self.anno_dir)

    def gen_labels(self) -> None:
        """
        Generates label files for APViT.
        """
        super().gen_labels()

        for _set in self.sets:
            self._gen_labels(_set)
        
    def _gen_labels(self, _set: str) -> None:
        """
        Generates label files for a specific subset.

        Args:
            _set (str): The subset (e.g., 'train', 'val').
        """
        df = self.data[self.data[DF_COLS['set_col']] == _set]
        df = df.apply(
            lambda row : format_anno_row(row, _set), 
            axis=1,
            result_type='expand')

        out_path = self.anno_dir + f'{SET_ALIASES[_set]}.txt'
        df.to_csv(out_path, sep=' ', index=False, header=False)

    def save_img(self, img: Image, row: pd.Series) -> None:
        """
        Saves the processed image for APViT.

        Args:
            img (Image): The processed image.
            row (pd.Series): The row containing the image data.
        """
        _set = row[DF_COLS['set_col']]
        filename = row[DF_COLS['filename_col']]

        f_filename = format_out_filename(filename, _set)
        out_path = os.path.join(self.img_dir, f_filename)

        img.save(out_path)


class EFHandler(StructHandler):
    """
    Handler for the EF output structure.

    Args:
        data (pd.DataFrame): The dataframe containing the dataset information.
        in_dir (str): The input directory.
        out_dir (str): The output directory.
    """
    def __init__(self, data : pd.DataFrame, in_dir : str, out_dir : str) -> None:
        super().__init__(data=data, struct_name='EF', in_dir=in_dir, out_dir=out_dir)

    def init_dirs(self) -> None:
        """
        Initializes the output directory structure for EF.
        """
        super().init_dirs()

        for _set in self.sets:
            self._init_dirs(_set)

    def _init_dirs(self, _set : str) -> None:
        """
        Initializes the output directory structure for a specific subset.

        Args:
            _set (str): The subset (e.g., 'train', 'val').
        """
        set_path = os.path.join(self.out_dir, SET_ALIASES[_set] + '/')

        for label in self.labels:
            label_path = os.path.join(set_path, label + '/')

            os.makedirs(label_path)

    def gen_labels(self) -> None:
        """
        Generates label files for EF.
        """
        super().gen_labels()
        print('EF output structure was selected. Skipping label generation',
              '(labels are already encoded in directory structure).')
        
    def save_img(self, img: Image, row: pd.Series) -> None:
        """
        Saves the processed image for EF.

        Args:
            img (Image): The processed image.
            row (pd.Series): The row containing the image data.
        """
        _set = row[DF_COLS['set_col']]
        filename = row[DF_COLS['filename_col']]
        label = row[DF_COLS['anno_col']]

        f_filename = format_out_filename(filename, _set)
        out_path = os.path.join(self.out_dir, SET_ALIASES[_set], label, f_filename)

        img.save(out_path)