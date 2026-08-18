# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import re

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def load_local_resnet_weights(backbone, path):
    """Map Hugging Face ResNet-50 weights to the official DETR backbone."""
    source_path = path
    if os.path.isdir(path):
        source_path = os.path.join(path, 'pytorch_model.bin')
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    target = backbone.state_dict()
    mapped = {}
    prefix = ''
    for key, value in source.items():
        target_key = None
        if key == 'resnet.embedder.embedder.convolution.weight':
            target_key = 'conv1.weight'
        elif key.startswith('resnet.embedder.embedder.normalization.'):
            suffix = key.removeprefix('resnet.embedder.embedder.normalization.')
            target_key = 'bn1.' + suffix
        else:
            match = re.fullmatch(
                r'resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.'
                r'(shortcut|layer)\.(?:(\d+)\.)?'
                r'(convolution|normalization)\.(.+)', key)
            if match:
                stage, block, branch, layer, kind, suffix = match.groups()
                base = f'layer{int(stage) + 1}.{block}.'
                if branch == 'shortcut':
                    target_key = base + f'downsample.{0 if kind == "convolution" else 1}.{suffix}'
                else:
                    target_key = base + f'{"conv" if kind == "convolution" else "bn"}{int(layer) + 1}.{suffix}'
        if target_key in target and target[target_key].shape == value.shape:
            mapped[target_key] = value
    expected = {key for key in target if not key.startswith('fc.')}
    missing = expected - set(mapped)
    if missing:
        raise RuntimeError(f'Incomplete ResNet mapping: {len(missing)} keys missing')
    backbone.load_state_dict(mapped, strict=False)
    return len(mapped)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 weights_path: str = ''):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        if weights_path:
            backbone_keys_loaded = load_local_resnet_weights(backbone, weights_path)
        else:
            backbone_keys_loaded = 0
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        self.backbone_keys_loaded = backbone_keys_loaded


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers,
                        args.dilation, args.backbone_weights)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
