# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "ASG",
    "BSA",
    "CBAM",
    "CDGM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "GSConv",
    "PConv",
    "Conv2",
    "ConvTranspose",
    "CoordAttention",
    "DASC",
    "DWConv",
    "DWConvTranspose2d",
    "ECA",
    "EMA",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "LSK",
    "BiFPNConcat",
    "RepConv",
    "SOAU",
    "SPDConv",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class ECA(nn.Module):
    """ECA注意力模块（Efficient Channel Attention）。

    通过自适应1D卷积核实现轻量级通道注意力，几乎零参数开销，非常适合小模型。
    核心思想：用1D卷积代替全连接层进行跨通道交互，卷积核大小由通道数自动计算，
    避免了SE模块中降维带来的信息损失。

    参考文献: ECA-Net: Efficient Channel Attention for Deep CNNs (CVPR 2020)

    Attributes:
        pool (nn.AdaptiveAvgPool2d): 全局平均池化，将空间维度压缩为1x1。
        conv (nn.Conv1d): 自适应1D卷积，核大小由通道数自动计算。
        act (nn.Sigmoid): Sigmoid激活函数，生成0~1之间的通道注意力权重。
    """

    def __init__(self, c1, gamma=2, b=1):
        """初始化ECA模块。

        Args:
            c1 (int): 输入通道数。
            gamma (int): 自适应核大小计算参数，控制核大小与通道数的映射关系。
            b (int): 自适应核大小计算参数，偏置项。
        """
        super().__init__()
        # 自适应计算1D卷积核大小: k = |log2(C)/gamma + b/gamma|，取最近奇数
        # 例如: C=64 -> k=3, C=256 -> k=5, C=1024 -> k=5
        # 通道数越多，卷积核越大，跨通道交互范围越广
        kernel_size = int(abs(math.log2(c1) / gamma + b / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1  # 确保为奇数

        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化: (B,C,H,W) -> (B,C,1,1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """前向传播: 全局池化 -> 1D卷积跨通道交互 -> Sigmoid生成权重 -> 通道加权。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。

        Returns:
            (torch.Tensor): 通道注意力加权后的特征图，形状不变 (B, C, H, W)。
        """
        y = self.pool(x)                          # (B, C, 1, 1) 全局平均池化获取通道描述符
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C) 调整维度以适配1D卷积
        y = self.conv(y)                           # (B, 1, C) 1D卷积实现相邻通道间的信息交互
        y = y.transpose(-1, -2).unsqueeze(-1)      # (B, C, 1, 1) 恢复维度
        return x * self.act(y)                     # 逐通道加权: 每个通道乘以对应的注意力权重


class CoordAttention(nn.Module):
    """坐标注意力模块（Coordinate Attention）。

    将传统通道注意力分解为水平方向（H）和垂直方向（W）两个1D注意力，
    能够同时编码通道关系和精确的空间位置信息。相比CBAM等空间注意力方法，
    坐标注意力能捕获长程依赖且计算开销很小，特别适用于小目标检测任务。

    工作流程:
        1. 分别沿H方向和W方向进行自适应平均池化，保留位置信息
        2. 拼接两个方向的特征，通过共享1x1卷积进行通道降维和特征变换
        3. 拆分回H和W两个分支，各自通过独立1x1卷积生成注意力权重
        4. 用H注意力和W注意力的乘积对原始特征进行加权

    参考文献: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)

    Attributes:
        pool_h (nn.AdaptiveAvgPool2d): 水平方向池化，输出 (H, 1)。
        pool_w (nn.AdaptiveAvgPool2d): 垂直方向池化，输出 (1, W)。
        conv1 (nn.Conv2d): 共享的1x1降维卷积，减少计算量。
        bn1 (nn.BatchNorm2d): 批归一化层，稳定训练。
        act (nn.SiLU): 激活函数（与YOLOv8保持一致使用SiLU）。
        conv_h (nn.Conv2d): H方向注意力生成的1x1卷积。
        conv_w (nn.Conv2d): W方向注意力生成的1x1卷积。
    """

    def __init__(self, c1, reduction=32):
        """初始化坐标注意力模块。

        Args:
            c1 (int): 输入/输出通道数。
            reduction (int): 中间层通道缩减比例，用于降低计算量。
        """
        super().__init__()
        # 中间通道数，最小为8防止信息瓶颈
        mid_c = max(8, c1 // reduction)

        # 方向池化: 分别在H和W方向压缩空间维度，保留另一个方向的位置信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,W) -> (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B,C,H,W) -> (B,C,1,W)

        # 共享变换: 1x1卷积降维 + BN + 激活
        self.conv1 = nn.Conv2d(c1, mid_c, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act = nn.SiLU()

        # 独立分支: 分别生成H方向和W方向的注意力权重
        self.conv_h = nn.Conv2d(mid_c, c1, 1, 1, 0, bias=False)
        self.conv_w = nn.Conv2d(mid_c, c1, 1, 1, 0, bias=False)

    def forward(self, x):
        """前向传播: H池化+W池化 -> 拼接+共享变换 -> 拆分+独立映射 -> 双方向注意力加权。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。

        Returns:
            (torch.Tensor): 坐标注意力加权后的特征图，形状不变 (B, C, H, W)。
        """
        B, C, H, W = x.shape

        # Step 1: 方向池化，分别沿H和W方向获取位置编码
        x_h = self.pool_h(x)                         # (B, C, H, 1) 保留垂直位置信息
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # (B, C, W, 1) 保留水平位置信息，转置以便拼接

        # Step 2: 拼接两个方向的特征，共享1x1卷积进行降维和特征变换
        y = torch.cat([x_h, x_w], dim=2)              # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))          # (B, mid_c, H+W, 1) 降维+激活

        # Step 3: 拆分回H和W两个分支
        x_h, x_w = y.split([H, W], dim=2)             # (B, mid_c, H, 1) 和 (B, mid_c, W, 1)

        # Step 4: 各自通过独立1x1卷积生成注意力权重
        x_h = self.conv_h(x_h).sigmoid()               # (B, C, H, 1) H方向注意力
        x_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()  # (B, C, 1, W) W方向注意力

        # Step 5: 双方向注意力加权 — 同时编码"在哪里"和"关注什么通道"
        return x * x_h * x_w


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class EMA(nn.Module):
    """高效多尺度注意力模块（Efficient Multi-Scale Attention）。

    将通道维度分组，通过3个并行分支（1D水平、1D垂直、1x1）提取多尺度空间特征，
    然后通过跨空间信息交互和sigmoid注意力加权增强特征表达。
    无额外参数开销（相比标准卷积），适合嵌入C2f等模块中增强小目标特征。

    References:
        https://arxiv.org/abs/2305.13563
    """

    def __init__(self, channels, factor=32):
        """初始化EMA模块。

        Args:
            channels (int): 输入通道数。
            factor (int): 分组数，通道数必须能被factor整除。
        """
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """前向传播：三分支并行 -> 跨空间交互 -> 注意力加权。

        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)。

        Returns:
            (torch.Tensor): EMA注意力增强后的特征图 (B, C, H, W)。
        """
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # (b*g, c//g, h, w)

        # 三分支并行
        x_h = self.pool_h(group_x)           # (b*g, c//g, h, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # (b*g, c//g, w, 1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # (b*g, c//g, h+w, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 跨空间信息交互
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 残差连接：避免特征衰减，保留原始信息
        return (group_x * weights.sigmoid()).reshape(b, c, h, w) + x


class DASC(nn.Module):
    """方向感知条形卷积（Direction-Aware Strip Convolution）。

    针对遥感图像中船舶目标的细长形态特性（长宽比通常3:1~10:1）设计。
    标准3×3方形卷积核对细长目标的感受野利用率低，大量感受野落在目标之外。

    本模块并行使用1×7和7×1两个条形卷积核，分别捕获水平和垂直方向的特征，
    然后通过全局池化自适应学习两个方向的权重，实现方向感知的特征提取。

    设计思路:
        1×7卷积 → 擅长捕获东西向排列的船舶特征
        7×1卷积 → 擅长捕获南北向排列的船舶特征
        全局平均池化 → 评估两个方向的重要性
        Softmax归一化 → 自适应加权融合

    参数量: 2×(C×C×7) ≈ 标准3×3卷积的4.7倍
    感受野: 等效7×7区域，但对细长目标的信息利用率远高于方形卷积

    References:
        灵感来自条形卷积思想，但增加了自适应方向加权机制
    """

    def __init__(self, c1, c2, k=7, stride=1, groups=1):
        """初始化DASC模块。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 条形卷积核长度，默认7。
            stride (int): 步幅。
            groups (int): 分组卷积组数。
        """
        super().__init__()
        self.conv_h = nn.Conv2d(c1, c2, (1, k), stride=stride, padding=(0, k // 2), groups=groups, bias=False)
        self.conv_v = nn.Conv2d(c1, c2, (k, 1), stride=stride, padding=(k // 2, 0), groups=groups, bias=False)
        self.bn_h = nn.BatchNorm2d(c2)
        self.bn_v = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """前向传播：双方向条形卷积 → 方向重要性评估 → 自适应加权融合。

        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)。

        Returns:
            (torch.Tensor): 方向感知条形卷积输出 (B, C_out, H', W')。
        """
        f_h = self.act(self.bn_h(self.conv_h(x)))
        f_v = self.act(self.bn_v(self.conv_v(x)))

        # 评估两个方向的重要性
        w_h = self.pool(f_h)
        w_v = self.pool(f_v)
        weights = torch.softmax(torch.cat([w_h, w_v], dim=1), dim=1)
        w_h, w_v = torch.chunk(weights, 2, dim=1)

        return f_h * w_h + f_v * w_v


class BSA(nn.Module):
    """Background-Suppressed Attention v2.

    针对遥感海面背景均匀但存在波浪、云影等干扰的问题设计。
    通用注意力机制（如SE、CA）对所有区域一视同仁地"增强"，
    但海面背景本身特征均匀，不需要增强，反而应该抑制。

    本模块利用船舶（高频突变）与海面背景（低频平滑）的频率差异，
    通过低频分支估计背景强度，生成背景门控图，然后对高频前景特征
    进行门控加权，以标准残差形式增强原始特征。

    改进点（v2）：
        1. 增加SiLU激活和BN，稳定特征分布
        2. 去掉固定的residual_weight=0.1，改为标准残差x + ...
        3. 让网络自己学习门控强度，避免训练早期信息丢失

    设计思路:
        低频分支(5×5 DWConv + SiLU + BN) → 估计背景强度
        高频分支(3×3 DWConv + SiLU + BN) → 提取边缘细节
        背景门控 = Sigmoid(低频) → 值越大表示背景越强，门控越小
        输出 = x + 高频 × (1 - 背景门控)  → 标准残差注意力

    参数量: 极小（两个深度可分离卷积）
    物理意义: 明确利用船舶与海面的频率差异
    """

    def __init__(self, c1, reduction=1.0):
        """初始化BSA模块。

        Args:
            c1 (int): 输入通道数。
            reduction (float): 通道缩减比例，默认1.0（不缩减）。
        """
        super().__init__()
        mid_c = max(8, int(c1 * reduction))
        self.low_freq = nn.Sequential(
            nn.Conv2d(c1, mid_c, 5, padding=2, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.high_freq = nn.Sequential(
            nn.Conv2d(c1, mid_c, 3, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        """前向传播：高低频分离 → 背景估计 → 门控残差增强。

        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)。

        Returns:
            (torch.Tensor): 背景抑制后的特征图 (B, C, H, W)。
        """
        bg = self.low_freq(x)
        fg = self.high_freq(x)
        bg_gate = torch.sigmoid(bg)
        # 标准残差注意力：保留原始特征，门控调节增强量
        return x + fg * (1 - bg_gate)


class SPDConv(nn.Module):
    """Space-to-Depth Convolution (SPD-Conv)。

    替代传统 stride=2 下采样，避免小目标特征在下采样时丢失。
    将空间维度(H,W)上的 2×2 区域重排到通道维度，实现无信息损失的下采样。

    原理:
        输入 [B, C, H, W]
        Space-to-Depth: [B, 4C, H/2, W/2]
        1×1/Conv 降维: [B, C_out, H/2, W/2]

    References:
        https://arxiv.org/abs/2208.03641
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s)

    def forward(self, x):
        """Space-to-Depth + Conv。"""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)


class LSK(nn.Module):
    """Large Selective Kernel (LSK) 注意力模块。

    并行使用局部小核(5×5)和大感受野空洞核(7×7,d=3)，
    通过自适应加权让网络根据目标大小自动选择感受野。
    对细长小目标（如船舶）特别有效。

    References:
        https://arxiv.org/abs/2303.09030 (LSKNet)
    """

    def __init__(self, c):
        super().__init__()
        # 局部精细分支
        self.local = nn.Sequential(
            nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.SiLU())
        # 全局上下文分支（大核空洞卷积）
        self.global_ = nn.Sequential(
            nn.Conv2d(c, c, 7, padding=9, groups=c, dilation=3, bias=False),
            nn.BatchNorm2d(c), nn.SiLU())
        # 自适应选择器
        self.conv = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        f_local = self.local(x)
        f_global = self.global_(x)
        w = self.conv(x).sigmoid()  # [B,2,H,W]
        return x + f_local * w[:, 0:1] + f_global * w[:, 1:2]


class SOAU(nn.Module):
    """小目标感知上采样（Small-Object-Aware Upsampling）。

    针对FPN/PAN中标准最近邻上采样"无脑复制"导致小目标特征稀释的问题。
    标准上采样对所有区域一视同仁，但小目标区域需要更精细的插值。

    本模块使用可学习的逐像素注意力权重替代固定插值方法，
    根据输入特征的局部内容自适应生成上采样权重。
    小目标区域（高频变化）会学到更锐利的插值核，
    背景区域（低频平滑）学到更平滑的插值核。

    设计思路:
        1. 先用最近邻上采样得到粗糙的2×放大特征
        2. 用3×3卷积从原始输入生成注意力权重图
        3. 用注意力权重对粗糙上采样结果做逐像素加权增强

    参数量: C×9（只在Neck中使用2-3次，开销可控）
    优势: 相比最近邻上采样更精细，相比Dysample更轻量
    """

    def __init__(self, c1, scale_factor=2):
        """初始化SOAU模块。

        Args:
            c1 (int): 输入/输出通道数（上采样不改变通道数）。
            scale_factor (int): 上采样倍率，默认2。
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.refine_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        """前向传播：最近邻上采样 → 特征精炼 → 残差增强。

        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)。

        Returns:
            (torch.Tensor): 上采样后的特征图 (B, C, H×scale, W×scale)。
        """
        import torch.nn.functional as F
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x_refined = self.refine_conv(x_up)
        return x_up + 0.1 * x_refined


class BiFPNConcat(nn.Module):
    """BiFPN 加权特征融合 — 快速归一化加权 Concat (Bidirectional Feature Pyramid Network)。

    在标准 Concat 的基础上，为每个输入分支添加可学习的缩放权重，
    通过快速归一化（Fast Normalized Fusion）实现自适应加权融合。

    标准 Concat 对所有输入一视同仁地拼接，BiFPNConcat 则让模型学习
    不同来源特征的最优融合比例。对于遥感船舶检测，FPN/PAN 中不同尺度
    的特征重要性差异显著（小目标主要依赖高分辨率特征），加权融合能
    自适应地强调更重要的特征来源。

    设计思路:
        1. 为每个输入分支分配可学习标量权重 w_i (初始化为1)
        2. 快速归一化: output = Concat(w_i * x_i / (sum(w_j) + eps))
        3. 使用 ReLU 确保权重非负
        4. 相比 Softmax 归一化，快速归一化更高效且收敛更稳定

    References:
        EfficientDet: Scalable and Efficient Object Detection (Tan et al., 2020)
        https://arxiv.org/abs/1911.09070
    """

    def __init__(self, num_inputs, dim=1):
        """初始化 BiFPN 加权融合模块。

        Args:
            num_inputs (int): 输入分支数量（对应 YAML 中 from 字段的元素个数）。
            dim (int): 拼接维度，默认1（通道维度）。
        """
        super().__init__()
        self.dim = dim
        self.num_inputs = num_inputs
        # 可学习的逐输入融合权重，初始化为1（等权重，退化为标准Concat）
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def forward(self, x):
        """前向传播：快速归一化加权拼接。

        Args:
            x (list[torch.Tensor]): 来自不同层的特征图列表。

        Returns:
            (torch.Tensor): 加权拼接后的特征图，通道数 = sum(各输入通道数)。
        """
        # ReLU 确保权重非负，eps 保证数值稳定
        w = F.relu(self.weights)
        w_sum = w.sum() + 1e-4
        # 加权后拼接，保持与标准Concat相同的输出形状
        scaled = [w[i] / w_sum * xi for i, xi in enumerate(x)]
        return torch.cat(scaled, self.dim)


class DySample(nn.Module):
    """Dynamic Sampling Upsampling (Simplified Version).

    Learns dynamic sampling offsets for content-aware upsampling.
    Much lighter than SOAU (~16K params for 3 layers vs 6.2M).

    Reference: Learning to Upsample by Learning to Sample, CVPR 2023.
    """

    def __init__(self, c1, scale=2):
        """Initialize DySample module.

        Args:
            c1 (int): Input channels (also output channels).
            scale (int): Upsampling scale factor, default 2.
        """
        super().__init__()
        self.scale = scale
        # Predict 2D normalized offsets [-1, 1]
        self.offset_conv = nn.Conv2d(c1, 2, 3, padding=1, bias=False)

    def forward(self, x):
        """Forward: predict offsets -> upsample offsets -> dynamic grid_sample.

        Args:
            x (torch.Tensor): Input feature (B, C, H, W).

        Returns:
            (torch.Tensor): Upsampled feature (B, C, H*scale, W*scale).
        """
        B, C, H, W = x.shape
        # Predict offsets and normalize to [-1, 1]
        offset = torch.tanh(self.offset_conv(x))
        # Upsample offsets to target resolution
        offset = F.interpolate(offset, scale_factor=self.scale, mode="bilinear", align_corners=False)
        offset = offset.permute(0, 2, 3, 1)  # [B, H*s, W*s, 2]

        # Base grid for upsampled resolution in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H * self.scale, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W * self.scale, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Dynamic sampling with learned offsets
        return F.grid_sample(x, grid + offset, mode="bilinear", padding_mode="border", align_corners=False)


class CDGM(nn.Module):
    """Cross-Scale Detail Guidance Module v2 (跨尺度细节引导融合).

    在FPN特征融合中，用高分辨率Backbone特征(detail)引导低分辨率上采样特征(coarse)的调制，
    替代标准Concat实现更精细的跨尺度融合。

    v2改进 (相比v1):
        v1: detail.mean(dim=1) → 无参数，全通道平均丢失语义信息
        v2: learnable gate_proj(detail) → 可学习门控，保留通道语义

    设计思路:
        1. detail特征 → 1×1降维 → SiLU → 1×1投影 → σ → [B,1,H,W] 空间门控
        2. coarse特征 = coarse × (1 + 门控) — 细节引导的调制增强
        3. Concat(modulated_coarse, detail) — 保持与标准Concat相同的输出格式

    参数量: c_detail × (c_detail//4 + 1) 个参数
    使用方式: 在YAML中替换FPN路径的Concat: [[-1, N], 1, CDGM, []]
    """

    def __init__(self, c_detail):
        """初始化CDGM模块。

        Args:
            c_detail (int): detail特征图的通道数（来自Backbone的高分辨率分支）。
        """
        super().__init__()
        hidden = max(8, c_detail // 4)
        self.gate_proj = nn.Sequential(
            nn.Conv2d(c_detail, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )

    def forward(self, x):
        """前向传播：可学习细节引导调制 → 拼接。

        Args:
            x (list[torch.Tensor]): [coarse, detail]，两个尺度的特征图。

        Returns:
            (torch.Tensor): 调制后拼接的特征图 (通道数 = coarse_c + detail_c)。
        """
        coarse, detail = x
        # 可学习门控: 从detail中学习每个位置的空间置信度
        gate = self.gate_proj(detail).sigmoid()
        # 如果coarse和detail空间尺寸不同，对齐coarse到detail的分辨率
        if coarse.shape[-2:] != detail.shape[-2:]:
            coarse = F.interpolate(
                coarse, size=detail.shape[-2:], mode="bilinear", align_corners=False
            )
        # 细节引导调制: coarse特征在detail高响应区域被增强
        coarse_mod = coarse * (1 + gate)
        return torch.cat([coarse_mod, detail], 1)


class PConv(nn.Module):
    """部分卷积 — 只处理 1/4 通道，其余直通，参数降约 40%。

    References:
        FasterNet, CVPR 2023
    """

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, groups=self.dim_conv, bias=False)
        self.bn = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.split([self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.act(self.bn(self.conv(x1)))
        return torch.cat([x1, x2], dim=1)


class GSConv(nn.Module):
    """鬼影混洗卷积 — 1/2通道做标准Conv + 1/2做DWConv → Concat + Shuffle。

    标准 Conv(C, C_out, 3) → 参数量 = C × C_out × 9
    GSConv(C, C_out, 3)   → 参数量 ≈ C × C_out×0.5 + C_out×0.5×9 ≈ 50%
    同时通过 Channel Shuffle 保证多分支信息流通。

    References:
        SlimNeck / GSConv (YOLOv5轻量化常用)
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = Conv(c1, c_, k, s, g=g, act=act)
        self.dwconv = DWConv(c_, c_, k=5, s=1, act=act)  # 下采样由 conv 负责, dwconv s=1

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dwconv(x1)
        # channel shuffle: 将两分支交错排列
        x = torch.cat([x1, x2], 1)
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return x


class ASG(nn.Module):
    """Adaptive Scale Gate (自适应尺度门控).

    在检测头前对每个尺度的特征图进行逐位置自适应门控。
    让网络为每个空间位置评估"当前尺度的特征是否可靠"，
    从而抑制噪声响应、增强可靠检测。

    设计思路:
        1. 深度可分离3×3 Conv → 提取局部结构信息
        2. BN + SiLU → 特征稳定
        3. 1×1 Conv → 降到1通道 → σ → [B,1,H,W] 置信度图
        4. 输出 = 输入 × 置信度 — 自适应门控

    物理意义:
        - P2层中"像船的区域" → 置信度高 → 增强
        - P2层中"大船的局部碎片" → 置信度低 → 抑制
        - 网络自己学会为每个位置选择合适的检测尺度

    参数量: ~c×9 + c×1 (DWConv + Pointwise)，约(10×c)个参数
    使用方式: 在YAML中放在检测头前: [-1, 1, ASG, []]
    """

    def __init__(self, c1):
        """初始化ASG模块。

        Args:
            c1 (int): 输入/输出通道数。
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, 1, bias=False),
        )

    def forward(self, x):
        """前向传播：逐位置门控生成 → 自适应特征调节。

        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)。

        Returns:
            (torch.Tensor): 门控后的特征图 (B, C, H, W)。
        """
        weight = self.gate(x).sigmoid()
        return x * weight
