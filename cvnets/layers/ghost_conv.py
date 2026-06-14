#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Optional, Union

import torch
from torch import Tensor, nn

from cvnets.layers import ConvLayer2d
from cvnets.layers.activation import build_activation_layer
from cvnets.layers.base_layer import BaseLayer
from cvnets.layers.normalization_layers import get_normalization_layer


class GhostConv2d(BaseLayer):
    """
    This class implements the Ghost Convolution module from
    `GhostNet: More Features from Cheap Operations <https://arxiv.org/abs/1911.11907>`_

    A Ghost module replaces a standard convolution by:
    1. A primary convolution producing a subset of output feature maps (intrinsic).
    2. A series of cheap linear transformations (depthwise convolutions) on the
       intrinsic feature maps to generate additional "ghost" feature maps.
    3. Concatenation of intrinsic and ghost feature maps.

    Args:
        opts: Command line arguments
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size for primary convolution
        stride: Stride for primary convolution. Default: 1
        padding: Padding for primary convolution. Default: None (auto)
        dilation: Dilation for primary convolution. Default: 1
        groups: Groups for primary convolution. Default: 1
        bias: Use bias. Default: False
        padding_mode: Padding mode. Default: "zeros"
        use_norm: Use normalization. Default: True
        use_act: Use activation. Default: True
        ratio: Ratio of intrinsic channels to total output channels.
               E.g., ratio=2 means half channels from primary conv, half from ghost. Default: 2
        dw_size: Kernel size for the depthwise cheap operations. Default: 3
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Optional[Union[int, tuple]] = None,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        use_norm: bool = True,
        use_act: bool = True,
        ratio: int = 2,
        dw_size: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        assert out_channels % ratio == 0, (
            f"out_channels ({out_channels}) must be divisible by ratio ({ratio})"
        )

        primary_channels = out_channels // ratio  # intrinsic feature maps
        ghost_channels = out_channels - primary_channels  # ghost feature maps

        # Primary convolution: produces intrinsic feature maps
        self.primary_conv = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=primary_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            use_norm=use_norm,
            use_act=use_act,
        )

        # Cheap operations: depthwise conv on intrinsic maps to produce ghost maps
        self.ghost_conv = nn.Sequential()
        if ghost_channels > 0:
            norm_layer = None
            if use_norm:
                norm_type = getattr(opts, "model.normalization.name")
                if norm_type == "batch_norm":
                    norm_type = "batch_norm_2d"
                norm_layer = get_normalization_layer(
                    opts=opts, num_features=ghost_channels, norm_type=norm_type
                )

            act_layer = None
            if use_act:
                act_layer = build_activation_layer(
                    opts, num_parameters=ghost_channels
                )

            self.ghost_conv.add_module(
                name="cheap_conv",
                module=ConvLayer2d(
                    opts=opts,
                    in_channels=primary_channels,
                    out_channels=ghost_channels,
                    kernel_size=dw_size,
                    stride=1,
                    padding=dw_size // 2,
                    groups=primary_channels,
                    bias=False,
                    padding_mode=padding_mode,
                    use_norm=False,
                    use_act=False,
                ),
            )
            if norm_layer is not None:
                self.ghost_conv.add_module(name="norm", module=norm_layer)
            if act_layer is not None:
                self.ghost_conv.add_module(name="act", module=act_layer)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.dw_size = dw_size
        self.primary_channels = primary_channels
        self.ghost_channels = ghost_channels

    def forward(self, x: Tensor) -> Tensor:
        intrinsic = self.primary_conv(x)
        if self.ghost_channels > 0:
            ghost = self.ghost_conv(intrinsic)
            x = torch.cat([intrinsic, ghost], dim=1)
        else:
            x = intrinsic
        return x

    def __repr__(self) -> str:
        return "{}(in={}, out={}, kernel_size={}, ratio={}, dw_size={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.primary_conv.block.conv.kernel_size,
            self.ratio,
            self.dw_size,
        )
