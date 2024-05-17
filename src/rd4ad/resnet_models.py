from typing import Any

from rd4ad import _modules
from torch import nn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
    Tensor,
    WeightsEnum,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    _ovewrite_named_param,
)


class EncoderBottleneckDecoderResNet(nn.Module):
    """
    The Teacher and Student are ResNet models modified to output layers at different stages.
    These are compared during training to enforce feature similarity.
    """

    def __init__(
        self,
        *,
        teacher_block: _modules.BasicBlock | _modules.Bottleneck,
        layers: list[int],
        teacher_weights: WeightsEnum,
        progress: bool,
        **kwargs: Any,
    ):
        super().__init__()
        bn_block, student_block = {
            _modules.BasicBlock: (_modules.AttnBasicBlock, _modules.DeBasicBlock),
            _modules.Bottleneck: (_modules.AttnBottleneck, _modules.DeBottleneck),
        }[teacher_block]

        self.teacher_encoder = _modules.FeatureResNet(teacher_block, layers, **kwargs)
        weights = teacher_weights.get_state_dict(progress=progress, check_hash=True)
        self.teacher_encoder.load_state_dict(weights)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()

        bn_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["groups", "width_per_group", "norm_layer"]
        }
        self.bottleneck = _modules.OneClassBottleneck(bn_block, layers[-1], **bn_kwargs)
        self.student_decoder = _modules.DeResNet(student_block, layers, **kwargs)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_layers = self.teacher_encoder(x)
        output_layers = self.student_decoder(self.bottleneck(input_layers))
        return input_layers, output_layers


def resnet18(
    *, weights=ResNet18_Weights.DEFAULT, progress: bool = True, **kwargs: Any
) -> EncoderBottleneckDecoderResNet:
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.BasicBlock,
        layers=[2, 2, 2, 2],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnet34(
    *, weights=ResNet34_Weights.DEFAULT, progress: bool = True, **kwargs: Any
) -> EncoderBottleneckDecoderResNet:
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.BasicBlock,
        layers=[3, 4, 6, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnet50(
    *, weights=ResNet50_Weights.DEFAULT, progress: bool = True, **kwargs: Any
) -> EncoderBottleneckDecoderResNet:
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 6, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnet101(
    *, weights=ResNet101_Weights.DEFAULT, progress: bool = True, **kwargs: Any
) -> EncoderBottleneckDecoderResNet:
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 23, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnet152(
    *, weights=ResNet152_Weights.DEFAULT, progress: bool = True, **kwargs: Any
) -> EncoderBottleneckDecoderResNet:
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 8, 36, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnext50_32x4d(
    *,
    weights=ResNeXt50_32X4D_Weights.DEFAULT,
    progress: bool = True,
    **kwargs: Any,
) -> EncoderBottleneckDecoderResNet:
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 6, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnext101_32x8d(
    *,
    weights=ResNeXt101_32X8D_Weights.DEFAULT,
    progress: bool = True,
    **kwargs: Any,
) -> EncoderBottleneckDecoderResNet:
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 23, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def resnext101_64x4d(
    *,
    weights=ResNeXt101_64X4D_Weights.DEFAULT,
    progress: bool = True,
    **kwargs: Any,
) -> EncoderBottleneckDecoderResNet:
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 23, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def wide_resnet50_2(
    *,
    teacher_weights=Wide_ResNet50_2_Weights.DEFAULT,
    progress: bool = True,
    **kwargs: Any,
) -> EncoderBottleneckDecoderResNet:
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 6, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )


def wide_resnet101_2(
    *,
    teacher_weights=Wide_ResNet101_2_Weights.DEFAULT,
    progress: bool = True,
    **kwargs: Any,
) -> EncoderBottleneckDecoderResNet:
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return EncoderBottleneckDecoderResNet(
        teacher_block=_modules.Bottleneck,
        layers=[3, 4, 23, 3],
        teacher_weights=weights,
        progress=progress,
        **kwargs,
    )
