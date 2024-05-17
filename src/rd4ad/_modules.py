# This file a simplification of https://github.com/manes-ese/anomaly-detection/tree/main/src/rd4ad
# This implementation just returns all 3 models in one and bases itself on torchvision==0.17.0

from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import Tensor, nn
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    WeightsEnum,
    conv1x1,
    conv3x3,
)


class AttnBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        # print("Attention:", self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.cbam = GLEAM(planes, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        # print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.cbam = GLEAM([int(planes * self.expansion/4),
        #                   int(planes * self.expansion//2),
        #                   planes * self.expansion], 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class OneClassBottleneck(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.cbam(x)
        # Upscale l1 and l2 to be same shape as l3
        l1, l2, l3 = x
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(l1))))))
        l2 = self.relu(self.bn3(self.conv3(l2)))
        feature = torch.cat([l1, l2, l3], 1)
        output = self.bn_layer(feature)
        # x = self.avgpool(feature_d)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class DeBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)


class DeBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)


class DeResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[DeBasicBlock, DeBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(DeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, DeBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[DeBasicBlock, DeBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        feature_a = self.layer1(x)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
        # feature_d = self.layer4(feature_c)  # 64*64*64->128*32*32

        # x = self.avgpool(feature_d)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [feature_c, feature_b, feature_a]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def deconv2x2(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=2,
        stride=stride,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class FeatureResNet(ResNet):
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        # feature_d = self.layer4(feature_c)

        return [feature_a, feature_b, feature_c]


def encoder_batchnorm_decoder_resnet(
    *,
    teacher_block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    teacher_weights: WeightsEnum,
    progress: bool,
    **kwargs: Any,
) -> tuple[FeatureResNet, OneClassBottleneck, DeResNet]:
    """
    Returns a tuple of the teacher encoder, batch norm layer, and student decoder.

    The Teacher and Student are ResNet models modified to output layers at different stages.
    These are compared during training to enforce feature similarity.
    """
    if teacher_block == BasicBlock:
        bn_block = AttnBasicBlock
        student_block = DeBasicBlock
    elif teacher_block == Bottleneck:
        bn_block = AttnBottleneck
        student_block = DeBottleneck
    else:
        raise ValueError("teacher_block must be BasicBlock or Bottleneck")

    teacher_encoder = FeatureResNet(teacher_block, layers, **kwargs)
    teacher_encoder.load_state_dict(
        teacher_weights.get_state_dict(progress=progress, check_hash=True)
    )
    teacher_encoder.requires_grad_(False)
    teacher_encoder.eval()

    bottleneck_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["groups", "width_per_group", "norm_layer"]
    }
    bottleneck = OneClassBottleneck(bn_block, layers[-1], **bottleneck_kwargs)
    student_decoder = DeResNet(student_block, layers, **kwargs)

    return teacher_encoder, bottleneck, student_decoder
