import itertools
import types
import typing
from functools import partial

import mlflow
import mlflow.pytorch
import numpy as np
import rd4ad.resnet_models
import segment_anything
import segment_anything.predictor as sam_pred
import SupContrast.networks.resnet_big
import torch
import torchvision.models.resnet
import torchvision.transforms.functional as F
from rd4ad.resnet_models import EncoderBottleneckDecoderResNet
from torch import nn
from torchvision.models.resnet import ResNet18_Weights, WeightsEnum

from anomaly_detection import config, util

T = typing.TypeVar("T")

weights_dict = {
    "imagenet": ResNet18_Weights.DEFAULT,
    "random": None,
}


def get_backbone_weights(dataset_name, model_name) -> WeightsEnum:
    class BackBone(WeightsEnum):
        DEFAULT = ResNet18_Weights.DEFAULT

        def get_state_dict(self, *args, **kwargs):
            if model_name == "resnet18_T-bottleneck":
                exp_name = f"embed_loss__{dataset_name}__resnet18_T-S__imagenet"
            else:
                exp_name = f"embed_loss__{dataset_name}__{model_name}__imagenet"

            run_ids = mlflow.search_runs(experiment_names=[exp_name]).run_id
            try:
                [run_id] = run_ids
            except Exception:
                raise ValueError(f"{exp_name} does not have exactly one run.")
            model = util.load_model_mlflow(run_id, train_pct=100)
            match model:
                case EncoderBottleneckDecoderResNet() if model_name == "resnet18_T-S":
                    return _reset_fc(self, model.teacher_encoder.state_dict())
                case EncoderBottleneckDecoderResNet() if model_name == "resnet18_T-bottleneck":
                    return model.state_dict()
                case SupConResNet():
                    return _reset_fc(self, model.encoder.state_dict())
                case _:
                    raise ValueError(f"Unknown model type: {type(model)}")

    return BackBone.DEFAULT


datasets = ["scavport", "vesselarchive", "dimo"]
unsupervised_models = ["resnet18_T-S", "resnet18_T-bottleneck", "resnet18_self-sup-con"]
combinations = [
    *itertools.product(datasets, unsupervised_models),
    ("scavport", "resnet18_sup-con"),
]
ResNet18_Weights.verify = lambda cls: cls  # Allow custom weights
for dataset_name, model_name in combinations:
    name = f"{model_name}@{dataset_name}"
    weights_dict[name] = get_backbone_weights(dataset_name, model_name)


def _reset_fc(weights: WeightsEnum, model):
    # We trained with 9 classes, but we don't use the last layers
    # So we just override them with ImageNet weights to allow the weights to load
    imagenet = WeightsEnum.get_state_dict(weights)
    model["fc.weight"] = imagenet["fc.weight"]
    model["fc.bias"] = imagenet["fc.bias"]
    return model


class TeacherBottleneck(nn.Module):
    def __init__(self, *, weights: WeightsEnum):
        super().__init__()
        self.model = rd4ad.resnet_models.resnet18()
        self.model.load_state_dict(weights.get_state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.bottleneck(self.model.teacher_encoder(x))


def get_sam_transform(img_size=512):
    sam = sam_pred.Sam(None, None, None)
    sam.image_encoder = types.SimpleNamespace(img_size=img_size)

    def transform(raw_img):
        """
        raw_img = cv2.imread(r"E:\vessel-archive\train\46455444943847.jpg")
        """
        raw_img = np.array(raw_img)
        img = sam_pred.ResizeLongestSide(img_size).apply_image(raw_img)
        img = torch.as_tensor(img)
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :]
        img = sam.preprocess(img).squeeze()
        return img

    return transform


class SamMaeEncoder(nn.Module):
    def __init__(
        self,
        weights: typing.Literal[
            "sam_vit_h_4b8939.pth",
            "sam_vit_l_0b3195.pth",
            "sam_vit_b_01ec64.pth",
        ] = "sam_vit_b_01ec64.pth",
        img_size=1024,
    ):
        """
        >>> weights = "sam_vit_l_0b3195.pth"
        """
        super().__init__()
        _, _, size, *_ = weights.split("_")
        sam_model = segment_anything.sam_model_registry[f"vit_{size}"](
            config.DATA_ROOT / weights
        ).to(config.DEVICE)

        enc = sam_model.image_encoder
        assert enc.img_size == 1024
        if img_size != 1024:
            downsample = nn.AdaptiveAvgPool2d((img_size // 16, img_size // 16))
            enc.pos_embed = nn.Parameter(
                downsample(enc.pos_embed.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            )

        self.encoder = nn.Sequential(enc, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))

        # import cv2
        # transform = get_sam_transform(img_size)
        # raw_img = cv2.imread(r"E:\vessel-archive\train\46455444943847.jpg")
        # input_image = transform(raw_img).to(config.DEVICE)
        # enc(input_image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SupConResNet(SupContrast.networks.resnet_big.SupConResNet):
    def __init__(
        self,
        name="resnet18",
        head="mlp",
        feat_dim=128,
        weights=ResNet18_Weights.DEFAULT,
    ):
        super().__init__(name, head, feat_dim)
        self.encoder = model_dict[f"{name}"](weights=weights)
        self.encoder.fc = nn.Identity()


model_dict = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet18_T-bottleneck": TeacherBottleneck,  # This is for embedding using a pre-trained TS model
    "resnet18_T-S": rd4ad.resnet_models.resnet18,
    "resnet18_sup-con": partial(SupConResNet, name="resnet18"),
    "resnet18_self-sup-con": partial(SupConResNet, name="resnet18"),
}


def crop_transform_only(transform):
    def forward(img: torch.Tensor) -> torch.Tensor:
        # https://github.com/pytorch/vision/blob/v0.17.0/torchvision/transforms/_presets.py#L57
        self = transform
        img = F.resize(
            img,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        img = F.center_crop(img, self.crop_size)
        # if not isinstance(img, Tensor):
        #     img = F.pil_to_tensor(img)
        # img = F.convert_image_dtype(img, torch.float)
        # img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    return forward


transform_dict = {
    "resnet18": ResNet18_Weights.DEFAULT.transforms(),
    "resnet18_crop_only": crop_transform_only(ResNet18_Weights.DEFAULT.transforms()),
    "sam1024": get_sam_transform(1024),
    "sam512": get_sam_transform(512),
    "sam224": get_sam_transform(224),
}


ModelType: typing.TypeAlias = typing.Literal[
    "resnet18",
    "resnet18_T-S",
    "resnet18_T-bottleneck",
    "resnet18_self-sup-con",
    "resnet18_sup-con",
]
WeightsType: typing.TypeAlias = typing.Literal[
    "imagenet",
    "random",
    "resnet18_T-S@dimo",
    "resnet18_T-S@scavport",
    "resnet18_T-S@vesselarchive",
    "resnet18_T-bottleneck@dimo",
    "resnet18_T-bottleneck@scavport",
    "resnet18_T-bottleneck@vesselarchive",
    "resnet18_self-sup-con@dimo",
    "resnet18_self-sup-con@scavport",
    "resnet18_self-sup-con@vesselarchive",
    "resnet18_sup-con@scavport",
]
TransformType: typing.TypeAlias = typing.Literal[
    "resnet18", "resnet18_crop_only", "sam1024", "sam512", "sam224"
]

if set(ModelType.__args__) != model_dict.keys():
    raise TypeError(f"ModelType: typing.TypeAlias = typing.Literal{list(model_dict)}")
if set(WeightsType.__args__) != weights_dict.keys():
    raise TypeError(
        f"WeightsType: typing.TypeAlias = typing.Literal{list(weights_dict)}"
    )
if set(TransformType.__args__) != transform_dict.keys():
    raise TypeError(
        f"TransformType: typing.TypeAlias = typing.Literal{list(transform_dict)}"
    )
