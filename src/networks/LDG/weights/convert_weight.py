"""
This script provides functionality to extract backbone weights from a APViT checkpoint file and save them in a specified format.
Functions:
    parse_args(): Parses command line arguments.
    main(): Main function that extracts backbone weights from a checkpoint and saves them.
Usage:
    Run the script with the required arguments to extract backbone weights:
        python convert_weight.py <checkpoint> <output> [--backbone_prefix <prefix>]
"""
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    parser.add_argument('--backbone_prefix', type=str, default='extractor')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict()
    prefix_length = len(args.backbone_prefix) + 1

    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith(args.backbone_prefix):
            #output_dict['state_dict'][key[prefix_length:]] = value
            output_dict[key[prefix_length:]] = value
            has_backbone = True
            
    if not has_backbone:
        raise Exception("Cannot find a extractor module in the checkpoint.")
    
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()