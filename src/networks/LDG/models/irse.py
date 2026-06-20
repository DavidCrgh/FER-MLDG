from mmcls.models.backbones import IRSE
from mmcv.runner import load_state_dict
import torch
from torch import nn


"""
Loads an IRSE instance from the given checkpoint weights.
"""
def load_base_irse(checkpoint_path : str = None) -> nn.Module:
    irse = IRSE(
        input_size=(112, 112),
        num_layers=50,
        mode='ir',
        return_index=[2],
        return_type='Tuple'
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        load_state_dict(irse, state_dict, strict=False)

    return irse