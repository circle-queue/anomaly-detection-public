import functools
import itertools
import pickle
import random
import tempfile
import typing
import warnings
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, TypeVar

import mlflow.pytorch
import numpy as np
import pandas as pd
import SupContrast.losses
import SupContrast.util
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder, default_loader

from anomaly_detection import config  # , models,

T = TypeVar("T")

pd.set_option("future.no_silent_downcasting", True)

Layers: TypeAlias = list[torch.Tensor]

if typing.TYPE_CHECKING:
    from patchcore.patchcore import PatchCore


class Evaluation(typing.NamedTuple):
    relative_path: str
    class_label: str
    metrics: dict[str, float]

    @staticmethod
    def batch(paths, labels, metrics: dict[str, list[float]]) -> list["Evaluation"]:
        return [
            Evaluation(path, label, dict(zip(metrics, vals)))
            for path, label, *vals in zip(paths, labels, *metrics.values(), strict=True)
        ]


class TrainTestDataLoaders(NamedTuple):
    train: DataLoader
    test: DataLoader


def collate_drop_labels(samples: list) -> tuple[torch.FloatTensor, None]:
    imgs, _ = zip(*samples)
    return torch.stack(imgs), None


def add_randaug(transforms, **kwargs):
    return nn.Sequential(
        torchvision.transforms.RandAugment(**kwargs),
        transforms,
    )


def add_double_aug(transforms):
    """
    Input: Tensor[Channel, Width, Height]
    Output: Tensor[2, Channel, Width, Height]
    """
    return lambda x: torch.stack((transforms(x), transforms(x)))


@functools.cache
def load_checkpoint_data(exp_name_or_run_id: str, train_pct: int) -> list[Evaluation]:
    # exp_name = "anomaly__scavport__resnet18_T-S__imagenet"
    # train_pct = 100
    run_id = run_id_from_exp_name(exp_name_or_run_id)
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    data = mlflow.artifacts.load_dict(f"{artifact_uri}/evals_{train_pct}.json")
    return [Evaluation(*sample.values()) for sample in data]


def run_id_from_exp_name(exp_name_or_run_id: str) -> str:
    try:
        mlflow.get_run(exp_name_or_run_id)
        run_id = exp_name_or_run_id
    except Exception:
        try:
            runs = mlflow.search_runs(experiment_names=[exp_name_or_run_id])
            [run_id] = runs.run_id
        except ValueError:
            raise ValueError(
                f"No exact match for {exp_name_or_run_id=} in {runs.run_id.tolist()=}"
            )
    return run_id


def idx_to_class(dataloader: DataLoader) -> dict[int, str]:
    return dict(map(reversed, dataloader.dataset.class_to_idx.items()))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageFolderSubset(ImageFolder):
    def __init__(self, *, classes: list[str], **kwargs):
        self.classes = sorted(classes)
        super().__init__(**kwargs)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        dir_classes, class_to_idx = super().find_classes(directory)
        missing_classes = set(self.classes) - set(dir_classes)
        assert not missing_classes, f"Missing classes: {missing_classes=}"
        return self.classes, {class_: idx for idx, class_ in enumerate(self.classes)}


def anomaly_maps(input_layers: Layers, output_layers: Layers) -> Layers:
    """
    Each layer is a different feature map from the encoder/decoder
    Therefore, they also have different sizes, and cannot simply be e.g. stacked or bulk processed
    """
    return [
        1 - F.cosine_similarity(in_layer, out_layer)
        for in_layer, out_layer in zip(input_layers, output_layers)
    ]


def filter_layers(
    input_layers,
    output_layers,
    keep_layers: list[Literal["layer1", "layer2", "layer3"]],
):
    layer_names = ["layer1", "layer2", "layer3"]
    assert keep_layers and all(
        name in layer_names for name in keep_layers
    ), f"{keep_layers = }"
    return zip(
        *[
            (in_layer, out_layer)
            for in_layer, out_layer, layer_name in zip(
                input_layers, output_layers, layer_names, strict=True
            )
            if layer_name in keep_layers
        ]
    )


def batch_layer_loss(
    input_layers: Layers,
    output_layers: Layers,
    pixel_agg_func=torch.Tensor.mean,
    layer_agg_func=sum,
) -> torch.Tensor:
    return layer_agg_func(
        pixel_agg_func(a_map.flatten(start_dim=1), dim=1)
        for a_map in anomaly_maps(input_layers, output_layers)
    )


def sup_con_loss(model, img, class_=None) -> torch.FloatTensor:
    """
    img, class_ = next(iter(test_dataloader))
    """
    # cat 2*n images
    img = torch.cat([img[:, 0], img[:, 1]], dim=0)

    img = img.to(config.DEVICE)
    # class_ = class_.to(config.DEVICE)

    projection = model(img)
    batch = projection.shape[0] // 2
    f1, f2 = torch.split(projection, [batch] * 2, dim=0)
    embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    return SupContrast.losses.SupConLoss()(embed, class_)


def concat_dataset(ds1: ImageFolder, ds2: ImageFolder) -> ImageFolder:
    ds1_targets, ds2_targets = (
        pd.Series(ds.targets).replace({v: k for k, v in ds.class_to_idx.items()})
        for ds in (ds1, ds2)
    )
    ds1_paths, ds2_paths = (list(zip(*ds.imgs))[0] for ds in (ds1, ds2))

    ds = ImageFolder.__new__(ImageFolder)

    to_copy = ["transform", "target_transform", "transforms", "loader"]
    for attr in to_copy:
        assert getattr(ds1, attr) == getattr(ds2, attr), f"{attr=}"
        setattr(ds, attr, getattr(ds1, attr))

    # Cannot do test_ds.targets, as there may be target transforms
    ds.classes = sorted(ds1.classes + ds2.classes)
    ds.class_to_idx = {label: idx for idx, label in enumerate(ds.classes)}
    assert len(ds.classes) == len(ds.class_to_idx), "Duplicate class in concat_dataset"
    ds.targets = (
        pd.concat([ds1_targets, ds2_targets])
        .replace(ds.class_to_idx)
        .astype(int)
        .tolist()
    )
    ds.imgs = ds.samples = [
        (path, target) for path, target in zip(ds1_paths + ds2_paths, ds.targets)
    ]
    return ds


def gaussian(tsr: torch.Tensor, kernel_size: int = 33, sigma: float = 4):
    # We use torchvision to support batches, whereas RD4AD used scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    # Scipy: "The Gaussian kernel will have size 2*radius + 1 along each axis. If radius is None, the default radius = round(truncate * sigma) will be used."
    # truncate is by default 4 and RD4AD used sigma=4, therefore the kernel size is 2*4*4 + 1=33
    _gaussian = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    )
    """Confirm that the two methods are almost equivalent:
    >>> import PIL, numpy
    >>> img = PIL.Image.open(Path(r'E:\scav-port-data\dev\outlier\105022.jpg')).convert('L')
    >>> a = numpy.array(gaussian(img), dtype='float64')
    >>> b = scipy.ndimage.gaussian_filter(img, sigma=4).astype('float64')
    >>> assert (a-b).max() <= 3
    """
    return _gaussian(tsr)


def upscaled_anomaly_maps(
    input_layers: Layers,
    output_layers: Layers,
    out_size: int = 224,
    multiply: bool = True,
):
    a_maps = [
        F.interpolate(
            torch.unsqueeze(a_map, dim=1),
            size=out_size,
            mode="bilinear",
            align_corners=True,
        )
        for a_map in anomaly_maps(input_layers, output_layers)
    ]
    return torch.stack(a_maps).prod(dim=0) if multiply else sum(a_maps)


@functools.cache
def load_checkpoint_df(
    exp_name_or_run_id: str,
    test_dataloader: DataLoader,
    train_pcts: list[int] | None = None,
):
    pcts = train_pcts or config.CKP_PCTS
    df = pd.concat(
        [
            pd.DataFrame(load_checkpoint_data(exp_name_or_run_id, train_pct)).assign(
                train_pct=train_pct
            )
            for train_pct in pcts
        ],
        ignore_index=True,
    )
    df[["path", "target"]] = test_dataloader.dataset.imgs * len(pcts)
    assert len(df[["class_label", "target"]].value_counts()) == df.class_label.nunique()
    df = pd.concat([df, pd.DataFrame(df.metrics.tolist())], axis=1)
    return df


def log_model(model: nn.Module, dst: str = "model") -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if type(model).__name__ == "PatchCore":
            save_patchcore(model)
        else:
            mlflow.pytorch.log_model(model, dst)


@functools.cache
def load_model_cached(exp_name_or_run_id: str, train_pct: int) -> nn.Module:
    run_id = run_id_from_exp_name(exp_name_or_run_id)
    return load_model_mlflow(run_id, train_pct)


def load_model_mlflow(run_id: str, train_pct: int) -> nn.Module:
    """
    >>> run_id = "a30017727f7d4989a1de28dbd52b3e95"
    """
    exp_name = mlflow.get_experiment(mlflow.get_run(run_id).info.experiment_id).name
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    disk_cache = (
        Path.home() / f".cache/{artifact_uri.replace(':', '')}/model_{train_pct}"
    )
    disk_cache.parent.mkdir(parents=True, exist_ok=True)

    if "patchcore" not in exp_name:
        if disk_cache.exists():
            model = mlflow.pytorch.load_model(disk_cache)
        else:
            model = mlflow.pytorch.load_model(
                f"{artifact_uri}/model_{train_pct}", disk_cache.parent
            )
        return model

    # Load patchcore (disk cached)
    import patchcore.patchcore

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri

    device = {"device": torch.device(config.DEVICE)}
    mlflow.artifacts.download_artifacts(
        f"{artifact_uri}/model_100", dst_path=disk_cache.parent
    )
    self = patchcore.patchcore.PatchCore(**device)
    with open(self._params_file(disk_cache, ""), "rb") as load_file:
        patchcore_params = pickle.load(load_file)

    backbone = mlflow.pytorch.load_model(f"{disk_cache}/backbone")
    self.load(**patchcore_params, **device, backbone=backbone)
    self.anomaly_scorer.load(disk_cache, "")
    return self


def try_float(x):
    try:
        return float(x)
    except Exception:
        return x


def load_dev_model(exp_name_or_run_id):
    run_id = run_id_from_exp_name(exp_name_or_run_id)
    return load_model_mlflow(run_id, 100)


def load_best_hyperparams(
    exp_name: str, direction: Literal["minimize", "maximize"], target_metric: str
) -> dict:
    """
    >>> exp_name = "embed_loss__scavport__resnet18_T-S__imagenet"
    >>> target_metric = "eval_loss"
    >>> direction = "minimize"
    """
    exp_name_optuna = f"optuna_{exp_name}"
    exp = mlflow.get_experiment_by_name(exp_name_optuna)
    assert exp, f"Experiment not found: {exp_name_optuna}"
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    argbest = {"maximize": "argmax", "minimize": "argmin"}[direction]
    best_idx = runs_df[f"metrics.{target_metric}"].agg(argbest)
    best_params = (
        runs_df.loc[best_idx]
        .filter(regex="params")
        .rename(lambda x: x.replace("params.", ""))
        .map(try_float)
        .to_dict()
    )
    return best_params


def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def with_paths(dataloader: DataLoader) -> list[tuple[typing.Any, list[str]]]:
    """Adds an output to the dataloader iterator, containing the paths of the images in the batch."""
    flat_paths = (path for path, _class in dataloader.dataset.imgs)

    # https://docs.python.org/3/library/itertools.html
    batched_paths = []
    while batch := list(itertools.islice(flat_paths, dataloader.batch_size)):
        batched_paths.append(batch)

    return zip(dataloader, batched_paths, strict=True)


def save_patchcore(model: "PatchCore"):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dst = Path(tmp) / "model_100"
        tmp_dst.mkdir()
        model.backbone.name = getattr(
            model.backbone, "name", model.backbone.__class__.__name__
        )
        model.save_to_path(tmp_dst)
        mlflow.log_artifact(tmp_dst)
        mlflow.pytorch.log_model(model.backbone, artifact_path="model_100/backbone")
