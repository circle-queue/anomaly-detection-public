import dataclasses
import itertools
import math
import os
import statistics
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import mlflow
import mlflow.pytorch
import sklearn.metrics
import torch
import torchmetrics
import tqdm.auto as tqdm
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

import optuna
from anomaly_detection import config, models, util
from anomaly_detection.params import (
    Params,
    PatchCoreParams,
    SupConParams,
    TSAnomalyParams,
    TSBackBoneParams,
)
from anomaly_detection.util import Evaluation
from optuna.integration.mlflow import MLflowCallback

if TYPE_CHECKING:
    from rd4ad.resnet_models import EncoderBottleneckDecoderResNet
    from SupContrast.networks.resnet_big import SupConResNet

RunId = str


def seed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        util.setup_seed(111)
        return func(*args, **kwargs)

    return wrapper


class Hooks(list[Callable]):
    """
    A list of functions to be called at a certain point in the training loop
    Each hook has side effects/mutates
    """

    def __call__(self) -> list[None | Any]:
        return [hook() for hook in self]


@dataclass
class ModelBuilder:
    """
    Simplifies modification of the train loop

    ### Example:
        run_id = ModelBuilder().add_optimizer(optimizer) (...) .run()
    """

    exp_name: str = ""
    dev: bool = True

    model: nn.Module = None
    loss_fn: _Loss = None
    metrics: list[torchmetrics.Metric] = None

    progress: tqdm.tqdm = None

    train_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    params: dict[str, Any] = field(default_factory=dict)

    # All hooks affect TRAINING only
    pre_batch_hooks: Hooks[Callable] = field(default_factory=Hooks)
    post_batch_hooks: Hooks[Callable] = field(default_factory=Hooks)
    post_loop_hooks: Hooks[Callable] = field(default_factory=Hooks)
    loss_hooks: Hooks[Callable] = field(default_factory=Hooks)

    evals: list[Evaluation] = field(default_factory=list)

    def train_batch(self) -> torch.FloatTensor:
        raise NotImplementedError("No train step added")

    def test(self) -> float:
        raise NotImplementedError("No test added")

    @seed
    def run(self) -> "float":  # Returns test result
        """Once all the hooks are added, we can execute the main epoch loop"""
        if not self.dev:
            self.model = util.load_dev_model(self.exp_name)
            self.train = self.post_loop_hooks
            self.exp_name = "test_" + self.exp_name

        if run := mlflow.active_run():
            assert mlflow.get_experiment(run.info.experiment_id).name.startswith('optuna_')
            self.train()
            test_score = self.test()
        else:
            assert not self.exp_name.startswith("optuna_")
            mlflow.set_experiment(self.exp_name)
            with mlflow.start_run():
                self.train()
                test_score = self.test()
        return test_score

    @seed
    def train(self):
        """A typical train loop. This is not used by PatchCore"""
        mlflow.log_params(self.params)
        print("Evaluating at steps", [f"{pct}%" for pct in config.CKP_PCTS])
        for pct in config.CKP_PCTS:
            loops_todo = int(self.progress.total * pct / 100) - self.progress.n
            for _ in range(loops_todo):
                self.pre_batch_hooks()
                util.setup_seed(  # Ensure pre- and post hooks do not affect training
                    self.progress.n
                )
                batch_loss = self.train_batch().item()
                self.post_batch_hooks()
                self.progress.update()
                self.progress.set_postfix_str(f"{batch_loss = :.4f}", refresh=False)
            self.post_loop_hooks()
        return self

    @seed
    def checkpoint_evaluations(self):
        def func():
            self.test()
            mlflow.log_dict(
                [e._asdict() for e in self.evals],
                artifact_file=f"evals_{config.CKP_PCTS[inc()]}.json",
            )

        inc = self._inc_func()
        self.post_loop_hooks.append(func)
        return self

    @seed
    def checkpoint_model(self):
        inc = self._inc_func()
        self.post_loop_hooks.append(
            lambda: util.log_model(self.model, f"model_{config.CKP_PCTS[inc()]}")
        )
        return self

    @seed
    def add_model(self, model_name: models.ModelType, weights: models.WeightsType):
        self.model = models.model_dict[model_name](
            weights=models.weights_dict[weights]
        ).to(config.DEVICE)
        self.exp_name += f"__{model_name}__{weights}"
        return self

    @seed
    def add_task__teacher_student_finetune(self, cfg: TSBackBoneParams):
        self._set_n_samples(cfg.n_samples)
        self.exp_name = "embed_loss__" + self.exp_name
        self.params = dataclasses.asdict(cfg)
        self.model: "EncoderBottleneckDecoderResNet"
        self.model.teacher_encoder.requires_grad_(True)
        params = [
            {
                "params": self.model.teacher_encoder.layer1.parameters(),
                "lr": cfg.lr_layer1,
            },
            {
                "params": self.model.teacher_encoder.layer2.parameters(),
                "lr": cfg.lr_layer2,
            },
            {
                "params": self.model.teacher_encoder.layer3.parameters(),
                "lr": cfg.lr_layer3,
            },
            {"params": self.model.bottleneck.parameters(), "lr": cfg.lr_bottleneck},
            {"params": self.model.student_decoder.parameters(), "lr": cfg.lr_student},
        ]
        self.add_optimizer_adam(params, cfg.beta1, cfg.beta2)
        self.add_scheduler_1cycle(cfg.lr_factor_1cycle)
        train_iter = util.cycle(self.train_dataloader)

        def _train_batch() -> torch.FloatTensor:
            self.model.train()
            imgs, _classes = next(train_iter)
            input_layers, output_layers = self.model(imgs.to(config.DEVICE))
            loss = util.batch_layer_loss(input_layers, output_layers).mean()
            loss.backward()
            return loss

        def test() -> float:
            self.model.teacher_encoder.train()
            self.model.eval()
            idx_to_class = util.idx_to_class(self.test_dataloader)

            self.evals = []
            with torch.no_grad():
                for (img, classes), paths in util.with_paths(self.test_dataloader):
                    input_layers, output_layers = self.model(img.to(config.DEVICE))
                    losses = util.batch_layer_loss(input_layers, output_layers).tolist()
                    class_labels = map(idx_to_class.get, classes.tolist())
                    metrics = {"loss": losses}
                    self.evals.extend(Evaluation.batch(paths, class_labels, metrics))
            avg_loss = statistics.mean(
                [sample.metrics["loss"] for sample in self.evals]
            )
            mlflow.log_metric("eval_loss", avg_loss)
            return avg_loss

        self.train_batch = _train_batch
        self.test = test
        return self

    @seed
    def add_task__teacher_student_anomaly(self, cfg: TSAnomalyParams):
        self._set_n_samples(cfg.n_samples)
        self.exp_name = "anomaly__" + self.exp_name
        self.params = dataclasses.asdict(cfg)
        self.model: "EncoderBottleneckDecoderResNet"
        self.model.teacher_encoder.requires_grad_(False)
        params = [
            {"params": self.model.bottleneck.parameters(), "lr": cfg.lr_bottleneck},
            {"params": self.model.student_decoder.parameters(), "lr": cfg.lr_student},
        ]
        self.add_optimizer_adam(params, cfg.beta1, cfg.beta2)
        self.add_scheduler_1cycle(cfg.lr_factor_1cycle)
        train_iter = util.cycle(self.train_dataloader)

        def _train_batch() -> torch.FloatTensor:
            self.model.train()
            self.model.teacher_encoder.eval()
            imgs, _classes = next(train_iter)
            input_layers, output_layers = util.filter_layers(
                *self.model(imgs.to(config.DEVICE)),
                keep_layers=cfg.layers_to_extract_from.split(","),
            )
            loss = util.batch_layer_loss(input_layers, output_layers).mean()
            loss.backward()
            return loss

        def test() -> float:
            self.model.eval()
            idx_to_class = util.idx_to_class(self.test_dataloader)

            self.evals = []
            with torch.no_grad():
                for (imgs, classes), paths in util.with_paths(self.test_dataloader):
                    input_layers, output_layers = util.filter_layers(
                        *self.model(imgs.to(config.DEVICE)),
                        keep_layers=cfg.layers_to_extract_from.split(","),
                    )
                    losses = util.batch_layer_loss(input_layers, output_layers).tolist()

                    anomaly_maps = util.gaussian(
                        util.upscaled_anomaly_maps(input_layers, output_layers)
                    )
                    maxes = torch.amax(anomaly_maps, dim=(1, 2, 3)).tolist()
                    means = torch.mean(anomaly_maps, dim=(1, 2, 3)).tolist()

                    class_labels = map(idx_to_class.get, classes.tolist())
                    metrics = {
                        "loss": losses,
                        "anomaly_max": maxes,
                        "anomaly_mean": means,
                    }
                    self.evals.extend(Evaluation.batch(paths, class_labels, metrics))
            is_outlier = self.test_dataloader.dataset.is_outlier
            scores = [
                sample.metrics[f"anomaly_{cfg.pixel_agg_func}"] for sample in self.evals
            ]
            roc_auc_score = sklearn.metrics.roc_auc_score(is_outlier, scores)
            mlflow.log_metric("anomaly_auc", roc_auc_score)
            return roc_auc_score

        self.train_batch = _train_batch
        self.test = test
        return self

    @seed
    def add_task__self_supervised_contrast_finetune(self, cfg: SupConParams):
        self.train_dataloader.collate_fn = util.collate_drop_labels
        self.add_task__supervised_contrast_finetune(cfg)
        return self

    @seed
    def add_task__supervised_contrast_finetune(self, cfg: SupConParams):
        self._set_n_samples(cfg.n_samples)
        self.exp_name = "embed_loss__" + self.exp_name
        self.params = dataclasses.asdict(cfg)
        self.model: "SupConResNet"
        params = [
            {"params": self.model.encoder.layer1.parameters(), "lr": cfg.lr_layer1},
            {"params": self.model.encoder.layer2.parameters(), "lr": cfg.lr_layer2},
            {"params": self.model.encoder.layer3.parameters(), "lr": cfg.lr_layer3},
            {"params": self.model.encoder.layer4.parameters(), "lr": cfg.lr_layer4},
            {"params": self.model.head.parameters(), "lr": cfg.lr_projector},
        ]
        self.add_optimizer_adam(params, cfg.beta1, cfg.beta2)
        self.add_scheduler_1cycle(cfg.lr_factor_1cycle)
        train_iter = util.cycle(self.train_dataloader)

        for ds in [self.train_dataloader.dataset, self.test_dataloader.dataset]:
            ds.transform = util.add_double_aug(util.add_randaug(ds.transform))

        def _train_batch() -> torch.FloatTensor:
            self.model.train()
            imgs, classes = next(train_iter)
            loss = util.sup_con_loss(self.model, imgs, classes).mean()
            loss.backward()
            return loss

        def test() -> None:
            self.model.eval()
            idx_to_class = util.idx_to_class(self.test_dataloader)

            self.evals = []
            with torch.no_grad():
                for (img, classes), paths in util.with_paths(self.test_dataloader):
                    labels = (
                        None
                        if self.train_dataloader.collate_fn == util.collate_drop_labels
                        else classes
                    )
                    losses = util.sup_con_loss(self.model, img, labels).cpu().tolist()
                    embeds = self.model(img[:, 0].to(config.DEVICE)).cpu().tolist()
                    class_labels = map(idx_to_class.get, classes.tolist())
                    metrics = {
                        "loss": losses,
                        "embeds": embeds,
                    }
                    self.evals.extend(Evaluation.batch(paths, class_labels, metrics))
            avg_loss = statistics.mean(
                [sample.metrics["loss"] for sample in self.evals]
            )
            mlflow.log_metric("eval_loss", avg_loss)
            return avg_loss

        self.train_batch = _train_batch
        self.test = test
        return self

    @seed
    def add_task__patchcore_anomaly(self, cfg: PatchCoreParams):
        # Import here since Windows errors, as Faiss is Linux exclusive
        from patchcore import common, patchcore, sampler

        device = {"device": torch.device(config.DEVICE)}
        nn_method = common.FaissNN(on_gpu=True, num_workers=os.cpu_count(), **device)

        featuresampler = sampler.ApproximateGreedyCoresetSampler(
            cfg.feature_retention_pct, **device
        )

        backbone = self.model
        if teacher_encoder := getattr(backbone, "teacher_encoder", None):
            backbone = teacher_encoder
        if encoder := getattr(backbone, "encoder", None):
            backbone = encoder
        backbone.requires_grad_(False)
        backbone.eval()

        self.model = patchcore.PatchCore(config.DEVICE)
        if isinstance(cfg.imagesize, str):
            cfg.imagesize = tuple(
                int(size) for size in cfg.imagesize.strip("()").split(", ")
            )
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=cfg.layers_to_extract_from.split(","),
            input_shape=cfg.imagesize,
            pretrain_embed_dimension=int(cfg.pretrain_embed_dimension),
            target_embed_dimension=int(cfg.target_embed_dimension),
            patchsize=int(cfg.patchsize),
            featuresampler=featuresampler,
            anomaly_scorer_num_nn=int(cfg.anomaly_scorer_num_nn),
            nn_method=nn_method,
            pixel_agg_func=cfg.pixel_agg_func,
            **device,
        )

        self.exp_name = "anomaly__" + self.exp_name + "_patchcore"
        self.params = dataclasses.asdict(cfg)

        # PatchCore expects dataloaders to only return img
        def collate_fn(samples: list[tuple]):
            imgs, _ = zip(*samples)
            return torch.stack(imgs)

        self.train_dataloader.collate_fn = collate_fn
        self.test_dataloader.collate_fn = collate_fn

        n_batches = math.ceil(cfg.n_samples / self.train_dataloader.batch_size)
        assert n_batches <= len(self.train_dataloader)

        def train():
            mlflow.log_params(self.params)
            torch.cuda.empty_cache()

            self.progress = tqdm.tqdm(
                itertools.islice(self.train_dataloader, n_batches),
                total=n_batches,
                unit="batches",
                desc=self.exp_name,
            )
            self.model.fit(self.progress)
            self.post_loop_hooks()
            return self

        def test() -> None:
            torch.cuda.empty_cache()

            scores, masks, *_ = self.model.predict(self.test_dataloader)
            scores = [ele.tolist() for ele in scores]  # Make json encodable
            masks = [ele.tolist() for ele in masks]
            paths, class_idxs = zip(*self.test_dataloader.dataset.imgs)
            class_labels = map(util.idx_to_class(self.test_dataloader).get, class_idxs)

            self.evals = [
                Evaluation(
                    path, label, metrics={"score": score}
                )  # , "segmentation": mask})
                for path, label, score, mask in zip(paths, class_labels, scores, masks)
            ]
            is_outlier = self.test_dataloader.dataset.is_outlier
            roc_auc_score = sklearn.metrics.roc_auc_score(is_outlier, scores)
            mlflow.log_metric("anomaly_auc", roc_auc_score)
            return roc_auc_score

        self.train = train
        self.test = test
        return self

    # These have label 1/True/Anomalous
    @seed
    def set_outliers(self, labels_inliers: list[str], labels_outlier: list[str]):
        test_ds = self.test_dataloader.dataset
        assert set(labels_outlier) | set(labels_inliers) == set(test_ds.classes)
        idx_is_outlier = {
            idx: label in labels_outlier for label, idx in test_ds.class_to_idx.items()
        }
        test_ds.is_outlier = [idx_is_outlier[cls_idx] for cls_idx in test_ds.targets]
        return self

    @seed
    def add_scavport_dataset(
        self,
        transform: Callable,
        batch_size: int,
        train_classes: list[str],
        test_classes: list[str],
    ):
        self.exp_name += "scavport"
        train_ds = util.ImageFolderSubset(
            transform=transform,
            classes=train_classes,
            root=config.SCAV_PORT_ROOT / "train",
        )
        test_ds = util.ImageFolderSubset(
            transform=transform,
            classes=test_classes,
            root=config.SCAV_PORT_ROOT / ("dev" if self.dev else "test"),
        )
        self._set_dataloaders(train_ds, test_ds, batch_size=batch_size)
        return self

    @seed
    def add_vessel_archive_dataset(self, transform: Callable, batch_size: int):
        assert self.dev, 'No "test" set for vessel archive'
        self.exp_name += "vesselarchive"
        ds_kwargs = dict(transform=transform, root=config.VESSEL_ARCHIVE_ROOT)
        train_ds = util.ImageFolderSubset(**ds_kwargs, classes=["train"])
        test_ds = util.ImageFolderSubset(
            **ds_kwargs, classes=["dev" if self.dev else "test"]
        )
        self._set_dataloaders(train_ds, test_ds, batch_size=batch_size)
        return self

    @seed
    def add_condition_dataset(
        self,
        sub_dataset: typing.Literal[
            "images-scavengeport-overview",
            "lock-condition",
            "ring-condition",
            "ring-surface-condition",
        ],
        transform: Callable,
        batch_size: int,
        train_classes: list[str],
        test_classes: list[str],
    ):
        self.exp_name += f"condition-{sub_dataset}"
        train_ds = util.ImageFolderSubset(
            transform=transform,
            classes=train_classes,
            root=config.CONDITION_ROOT / sub_dataset / "train",
        )
        test_ds = util.ImageFolderSubset(
            transform=transform,
            classes=test_classes,
            root=config.CONDITION_ROOT / sub_dataset / ("dev" if self.dev else "test"),
        )
        self._set_dataloaders(train_ds, test_ds, batch_size=batch_size)
        return self

    def add_dimo_ds(self, transform: Callable, batch_size: int):
        raise NotImplementedError("TODO")

    @seed
    def add_optimizer_adam(self, params, beta1, beta2):
        self.optimizer = torch.optim.Adam(params, betas=(beta1, beta2))
        self.pre_batch_hooks.append(lambda: self.optimizer.zero_grad())
        self.post_batch_hooks.append(lambda: self.optimizer.step())
        return self

    @seed
    def add_scheduler_1cycle(self, factor):
        max_lr = [p["lr"] * factor for p in self.optimizer.param_groups]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lr, total_steps=self.progress.total
        )
        self.post_batch_hooks.append(lambda: self.scheduler.step())
        return self

    @staticmethod
    @seed
    def run_optuna(
        build_model_func: Callable[[Params], "ModelBuilder"],
        params_cls: typing.Type[Params],
        optuna_direction: typing.Literal["maximize", "minimize"],
        optuna_metric_name: typing.Literal["eval_loss", "anomaly_auc"],
        n_trials: int = 100,
        n_jobs: int = 1,
        trial_overrides: dict = {},
        timeout: int = None,
    ):
        def build_cfg(trial: optuna.Trial) -> Params:
            """The trials object will suggest values for each parameter"""
            return params_cls(
                **{
                    key: (trial_overrides.get(key) or annot.__metadata__[0])(trial)
                    for key, annot in params_cls.__annotations__.items()
                }
            )

        mlflow_callback = MLflowCallback(
            mlflow_kwargs={"nested": True},
            tracking_uri=config.MLFLOW_URI,
            metric_name=optuna_metric_name,
        )

        @mlflow_callback.track_in_mlflow()
        def objective(trial: optuna.Trial) -> float:
            torch.cuda.empty_cache()
            fail_score = 0 if optuna_direction == "maximize" else 999
            fail_when = ["outofmemory", "out of memory", "zero features extracted"]
            cfg = build_cfg(trial)
            try:
                return build_model_func(cfg).run()
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                print(f"Failing due to {error = !s}")
                return fail_score
            except Exception as error:
                if any(error_msg in str(error).lower() for error_msg in fail_when):
                    print(f"Failing due to {error = !s}")
                    return fail_score
                raise

        # hack to get the experiment name
        exp_name = (
            "optuna_"
            + build_model_func(build_cfg(optuna.create_study().ask())).exp_name
        )
        mlflow.set_experiment(exp_name)
        optuna_db = config.OPTUNA_PATH_TEMPLATE.format(exp_name)
        study = optuna.create_study(
            storage=optuna_db,
            direction=optuna_direction,
            study_name=exp_name,
            load_if_exists=True,
        )
        trials_todo = n_trials - len(study.get_trials())
        if trials_todo <= 0:
            return
        with mlflow.start_run():
            mlflow.enable_system_metrics_logging()
            try:
                study.optimize(
                    objective,
                    timeout=timeout,
                    n_trials=trials_todo,
                    n_jobs=n_jobs,
                    gc_after_trial=True,
                    callbacks=[mlflow_callback],
                )
            finally:
                mlflow.log_artifact(optuna_db.replace("sqlite:///", ""))

    @seed
    def _inc_func(self):
        def func():
            nonlocal i
            i += 1
            return i

        # For PatchCore, we cannot perform iterative evaluation, so we skip to final eval
        # For test, we don't do training, so we skip to final eval
        is_test = not self.dev
        start = -1 if (is_test or "patchcore" in self.exp_name) else 0
        i = start - 1
        return func

    @seed
    def _set_n_samples(self, n_samples: int):
        self.progress = tqdm.trange(
            round(n_samples / self.train_dataloader.batch_size),
            unit="batches",
            desc=self.exp_name,
        )
        return self

    @seed
    def _set_dataloaders(self, train_ds, test_ds, **dl_kwargs):
        dl_kwargs["pin_memory"] = torch.cuda.is_available()
        self.train_dataloader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
        self.test_dataloader = DataLoader(test_ds, **dl_kwargs)
        return self
