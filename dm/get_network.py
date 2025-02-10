import pydoc
import warnings
from typing import Union

def get_network_from_plans( arch_kwargs, input_channels, output_channels,
                           allow_init=True, deep_supervision=False):
    network_class = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
    architecture_kwargs = dict(**arch_kwargs)
    arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"]
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)


    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network
if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        input_channels=1,
        output_channels=4,
    )
    data = torch.rand((1, 1, 304, 304))
    target = torch.rand(size=(1, 1, 512, 512))
    outputs = model(data) # this should be a list of torch.Tensor
#if __name__ == "__main__":
#    import torch
#
#    model = get_network_from_plans(
#        arch_kwargs={
#            'n_stages': 8, 
#            'features_per_stage': [32, 64, 128, 256, 512, 512, 512, 512], 
#            'conv_op': 'torch.nn.modules.conv.Conv2d', 
#            'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 
#            'strides': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], 
#            'n_conv_per_stage': [2, 2, 2, 2, 2, 2, 2, 2], 
#            
#            'n_conv_per_stage_decoder': [2, 2, 2, 2, 2, 2, 2], 
#            'conv_bias': True, 
#            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d', 
#            'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 
#            'dropout_op': None, 
#            'dropout_op_kwargs': None, 
#            'nonlin': 'torch.nn.LeakyReLU', 
#            'nonlin_kwargs': {'inplace': True}
#        },
#        input_channels=3,
#        output_channels=1,
#    )
#    data = torch.rand((1, 3, 304, 304))
#    target = torch.rand(size=(1, 1, 256, 256))
#    outputs = model(data) # this should be a list of torch.Tensor