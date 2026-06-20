from mmcls.models.backbones import IRSE
from mmcv.runner import load_state_dict
from .EfficientFace import LocalFeatureExtractor
from .modulator import Modulator

import torch
from torch import nn

class ModifiedLDG(IRSE):
    def __init__(self, input_size, num_layers, mode='ir', with_head=False, pretrained=None, return_index=(0, 1, 2), return_type='Tuple', num_classes=7, uses_ef_modules=True):
        super().__init__(
            input_size=input_size, 
            num_layers=num_layers, 
            mode=mode, 
            with_head=False, 
            pretrained=None, # We'll override the behavior for with_head and pretrained
            return_index=return_index, 
            return_type=return_type)
        
        # Store the flag for using EfficientFace modules
        self.uses_ef_modules = uses_ef_modules
        
        # NOTE: the modules intialized here use PyTorch's default Kaiming initialization, which is theoretically better
        # for networks that use (P)ReLU activations. However, the original IRSE layers use Xavier initialization.
        stage0_out_channels = 64
        stage1_out_channels = 128
        
        # Only initialize local and modulator modules if uses_ef_modules is True
        if self.uses_ef_modules:
            self.local = LocalFeatureExtractor(stage0_out_channels, stage1_out_channels, 1)  # The index parameter is unused
            self.modulator = Modulator(stage1_out_channels)
        
        if with_head:
            self.with_head = with_head
            
            if input_size[0] == 112:
                self.output_layer = nn.Sequential(
                                    nn.BatchNorm2d(512),
                                    nn.Dropout(),
                                    nn.Flatten(),
                                    nn.Linear(512 * 7 * 7, 1024),  # Intermediate layer
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(1024, num_classes),
                                    nn.BatchNorm1d(num_classes)
                                )
            else:
                self.output_layer = nn.Sequential(
                                    nn.BatchNorm2d(512),
                                    nn.Dropout(),
                                    nn.Flatten(),
                                    nn.Linear(512 * 14 * 14, 1024),  # Intermediate layer
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(1024, num_classes),
                                    nn.BatchNorm1d(num_classes)
                                )
                
        if pretrained is not None:
            self.init_weights(pretrained)
    
    """
    Override of IRSE init_weights method to load the entire body instead of just the first 3 stages.
    """
    def init_weights(self, pretrained):
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        stage_unit_nums = {
            34: (3, 4, 6, 3),
            50: (3, 4, 14, 3),
        }
        stage_num = stage_unit_nums[self.num_layers]

        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith('body.'):
                index = int(k.split('.')[1])
                if 0 <= index < stage_num[0]:
                    new_key = k.replace('body.', 'body.0.')
                elif stage_num[0] <= index < sum(stage_num[:2]):
                    new_key = f"body.1.{index-sum(stage_num[:1])}.{'.'.join(k.split('.')[2:])}"
                elif sum(stage_num[:2]) <= index < sum(stage_num[:3]):
                    new_key = f"body.2.{index-sum(stage_num[:2])}.{'.'.join(k.split('.')[2:])}"
                elif sum(stage_num[:3]) <= index < sum(stage_num[:4]):
                    new_key = f"body.3.{index-sum(stage_num[:3])}.{'.'.join(k.split('.')[2:])}"
                else:
                    new_key = k
            else:
                new_key = k
            new_state_dict[new_key] = v

        load_state_dict(self, new_state_dict, strict=False)

    def forward(self, x):
        if self.num_layers == 0:
            return x
        x = self.input_layer(x)
        
        x = self.body[0](x)
        
        if self.uses_ef_modules:
            x = self.modulator(self.body[1](x)) + self.local(x)
        else:
            x = self.body[1](x)
        
        for module in self.body[2:]:
            x = module(x)
        
        if self.with_head:
            x = self.output_layer(x)
        
        return x if self.return_type == 'Tensor' else (x,)


""" Loads an M-LDG instance from the given checkpoint weights. """
def load_base_mLDG(
        pretrained: str = None,
        checkpoint_path: str = None, 
        uses_ef_modules: bool = True, 
        num_classes: int = 7) -> nn.Module:
    
    mldg = ModifiedLDG(
        input_size=(112, 112),
        num_layers=50,
        mode='ir',
        with_head=True,
        pretrained=pretrained,
        return_index=[3],
        return_type='Tensor',
        num_classes=num_classes,
        uses_ef_modules=uses_ef_modules
    )
    
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu') 
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        load_state_dict(mldg, state_dict, strict=False)
    
    return mldg