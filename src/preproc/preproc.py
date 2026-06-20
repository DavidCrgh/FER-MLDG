"""
preproc.py

This script preprocesses the dataset for training, validation, or both. It supports
different output structures and image alignment using RetinaFace. The script reads
the dataset, initializes the output directory structure, generates labels, and
processes images if required.

Usage:
    python preproc.py <path> [--set {train,val,all}] [--out_struct {APVIT,EF}]
                      [--out_height HEIGHT] [--out_width WIDTH] [--out_dir DIR]
                      [--padding PADDING] [--no_img]
"""
import argparse

from lib.data_io import *
from lib.align import *
from lib.util import DF_COLS


def parseargs():
    parser = argparse.ArgumentParser(description="Preprocess the dataset for training, validation, or both.")

    parser.add_argument('path', type=str, default='./data/', help="Path to the dataset directory.")
    parser.add_argument('--set', choices=['train', 'val', 'all'], default='all', help="Subset of the dataset to process: 'train', 'val', or 'all'.")
    parser.add_argument('--out_struct', '-s', choices=['APVIT', 'EF'], default='APVIT', help="Output structure format: 'APVIT' or 'EF'.")
    parser.add_argument('--out_height', '-oh', default=HEIGHT_DEF, type=int, help="Height of the output images.")
    parser.add_argument('--out_width', '-ow', default=WIDTH_DEF, type=int, help="Width of the output images.")
    parser.add_argument('--out_dir', '-o', type=str, default='./out/', help="Directory to save the processed output.")
    parser.add_argument('--padding', '-p', type=float, default=PAD_DEF, help="Padding to apply to the images expressed as a percentage of the image's original size.")
    parser.add_argument('--no_img', action='store_true', help="Flag to skip image processing.")
    parser.add_argument('--filter_contempt', action='store_true', help="Flag to filter out the contempt class (label 7).")

    args = parser.parse_args()

    return args


def main():
    args = parseargs()

    # Add trailing slash to the path if it's missing
    if not args.path.endswith('/'):
        args.path += '/'

    df = None

    if args.set == 'all':
        df = read_dataset_heads(args.path, subset='train')
        df = df._append(read_dataset_heads(args.path, subset='val'))
    else:
        df = read_dataset_heads(args.path, subset=args.set)

    # Filter out the contempt class if the flag is set
    if args.filter_contempt:
        df = df[df[DF_COLS['anno_col']] != '7']

    if args.out_struct == 'APVIT':
        handler_class = APViTHandler
    else:
        handler_class = EFHandler

    struct_handler = handler_class(in_dir=args.path, out_dir=args.out_dir, data=df)
    struct_handler.create_struct()

    if not args.no_img:
        print('Aligning images using RetinaFace...')
        process_dataset(
            data=df,
            args=args,
            save_func=struct_handler.save_img
        )
    else:
        print('--no_image flag was passed. Skipping face alignment.')


if __name__ == '__main__':
    main()