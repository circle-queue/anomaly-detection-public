import functools
import typing
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Callable

import typer

from anomaly_detection.params import (
    Params,
    PatchCoreParams,
    SupConParams,
    TSAnomalyParams,
    TSBackBoneParams,
)

if TYPE_CHECKING:
    # Lazy import for startup cost
    from anomaly_detection.model_builder import ModelBuilder


def main(pipeline: list[Callable]):
    """
    The main function which accumulates all the sub-commands and executes them in order.

    E.g. if the user runs:
        ```
        python -m anomaly_detection \\
            dataloader_scavport --transform resnet18 --train_classes "[0, 1, 2]" --test_classes "[3, 4, 5]" \\
            add_model --model_name resnet18 --weights scavport \\
            run
        ```
    The `pipeline` argument will be:
        ```
        [dataloader_scavport, add_model, run]
        ```

    """
    from anomaly_detection.model_builder import ModelBuilder

    def build(steps: list[Callable[["ModelBuilder"], "ModelBuilder"]]) -> ModelBuilder:
        builder = ModelBuilder()
        for transform in steps:
            builder = transform(builder)
        return builder

    if pipeline[-1].__name__ != "_run_optuna":
        return build(pipeline)

    *pipeline, task_func, _run_optuna = pipeline
    assert task_func.__name__.startswith("add_task"), task_func.__name__
    # For optuna, we must re-initialize the state of the model builder
    # However, the model builder requires parameters from optuna to initialize
    # Therefore, we cannot initialize the model builder until we have the optuna params
    # To solve this, we need a function which inputs the optuna params and returns the initialized model builder

    def model_builder_func(cfg: Params) -> ModelBuilder:
        return task_func(build(pipeline), cfg)

    if TYPE_CHECKING:
        # For more details, see
        ModelBuilder.run_optuna
        # We don't directly call "ModelBuilder.run_optuna" here because the CLI configures the kwargs. See:
        run_optuna

    return _run_optuna(model_builder_func)


app = typer.Typer(
    no_args_is_help=True,
    result_callback=main,
    chain=True,
    pretty_exceptions_enable=False,
)


class UnsupervisedBackboneMethods(str, Enum):
    ReverseDistillation = "ReverseDistillation"
    SelfSupervisedContrast = "SelfSupervisedContrast"


class OptunaDirection(str, Enum):
    maximize = "maximize"
    minimize = "minimize"


class OptunaMetricName(str, Enum):
    eval_loss = "eval_loss"
    anomaly_auc = "anomaly_auc"


class AllBackboneMethods(str, Enum):
    SupervisedContrast = "SupervisedContrast"
    ReverseDistillation = "ReverseDistillation"
    SelfSupervisedContrast = "SelfSupervisedContrast"


class ModelType(str, Enum):
    resnet18 = "resnet18"
    resnet18_TS = "resnet18_T-S"
    resnet18_T_bottleneck = "resnet18_T-bottleneck"
    resnet18_supcon = "resnet18_sup-con"
    resnet18_selfsupcon = "resnet18_self-sup-con"


class WeightsType(str, Enum):
    imagenet = "imagenet"
    random = "random"
    resnet18_T_S_dimo = "resnet18_T-S@dimo"
    resnet18_T_S_scavport = "resnet18_T-S@scavport"
    resnet18_T_S_vessel_archive = "resnet18_T-S@vesselarchive"
    resnet18_T_bottleneck_dimo = "resnet18_T-bottleneck@dimo"
    resnet18_T_bottleneck_scavport = "resnet18_T-bottleneck@scavport"
    resnet18_T_bottleneck_vessel_archive = "resnet18_T-bottleneck@vesselarchive"
    resnet18_self_sup_con_dimo = "resnet18_self-sup-con@dimo"
    resnet18_self_sup_con_scavport = "resnet18_self-sup-con@scavport"
    resnet18_self_sup_con_vessel_archive = "resnet18_self-sup-con@vesselarchive"
    resnet18_sup_con_scavport = "resnet18_sup-con@scavport"


class TransformType(str, Enum):
    resnet18 = "resnet18"


class ConditionSubDataset(str, Enum):
    images_scavengeport_overview = "images-scavengeport-overview"
    lock_condition = "lock-condition"
    ring_condition = "ring-condition"
    ring_surface_condition = "ring-surface-condition"


@app.command()
def dataloader_scavport(
    transform: Annotated[TransformType, typer.Option()],
    train_classes: Annotated[list, typer.Option(parser=eval)],
    test_classes: Annotated[list, typer.Option(parser=eval)],
    batch_size: Annotated[int, typer.Option()] = 16,
):
    assert isinstance(train_classes, list) and isinstance(test_classes, list)

    def func(builder: "ModelBuilder"):
        import anomaly_detection.model_builder as build

        return builder.add_scavport_dataset(
            transform=build.models.transform_dict[transform],
            train_classes=train_classes,
            test_classes=test_classes,
            batch_size=batch_size,
        )

    return func


@app.command()
def dataloader_vesselarchive(
    transform: Annotated[TransformType, typer.Option()],
    batch_size: Annotated[int, typer.Option()] = 16,
):
    def func(builder: "ModelBuilder"):
        import anomaly_detection.model_builder as build

        return builder.add_vessel_archive_dataset(
            transform=build.models.transform_dict[transform], batch_size=batch_size
        )

    return func


@app.command()
def dataloader_condition(
    sub_dataset: Annotated[ConditionSubDataset, typer.Option()],
    transform: Annotated[TransformType, typer.Option()],
    train_classes: Annotated[list, typer.Option(parser=eval)],
    test_classes: Annotated[list, typer.Option(parser=eval)],
    batch_size: Annotated[int, typer.Option()] = 16,
):
    assert isinstance(train_classes, list) and isinstance(test_classes, list)

    def func(builder: "ModelBuilder"):
        import anomaly_detection.model_builder as build

        return builder.add_condition_dataset(
            sub_dataset=sub_dataset.value,
            transform=build.models.transform_dict[transform],
            train_classes=train_classes,
            test_classes=test_classes,
            batch_size=batch_size,
        )

    return func


@app.command()
def set_outliers(
    labels_inliers: Annotated[list, typer.Option(parser=eval)],
    labels_outliers: Annotated[list, typer.Option(parser=eval)],
):
    def func(builder: "ModelBuilder"):
        return builder.set_outliers(labels_inliers, labels_outliers)

    return func


@app.command()
def add_model(
    model_name: Annotated[ModelType, typer.Option()],
    weights: Annotated[WeightsType, typer.Option()],
):
    def func(builder: "ModelBuilder"):
        return builder.add_model(
            model_name=model_name,
            weights=weights,
        )

    return func


@app.command()
def set_n_samples(n_samples: int):
    def func(builder: "ModelBuilder"):
        return builder._set_n_samples(n_samples)

    return func


@app.command()
def add_task__teacher_student_finetune(cfg: Annotated[dict, typer.Option(parser=eval)]):
    def add_task(builder: "ModelBuilder", cfg):
        return builder.add_task__teacher_student_finetune(cfg)

    return _handle_cfg(cfg, TSBackBoneParams, add_task)


@app.command()
def add_task__teacher_student_anomaly(cfg: Annotated[dict, typer.Option(parser=eval)]):
    def add_task(builder: "ModelBuilder", cfg):
        return builder.add_task__teacher_student_anomaly(cfg)

    return _handle_cfg(cfg, TSAnomalyParams, add_task)


@app.command()
def add_task__patchcore_anomaly(cfg: Annotated[dict, typer.Option(parser=eval)]):
    def add_task(builder: "ModelBuilder", cfg):
        return builder.add_task__patchcore_anomaly(cfg)

    return _handle_cfg(cfg, PatchCoreParams, add_task)


@app.command()
def add_task__self_supervised_contrast_finetune(
    cfg: Annotated[dict, typer.Option(parser=eval)],
):
    def add_task(builder: "ModelBuilder", cfg):
        return builder.add_task__self_supervised_contrast_finetune(cfg)

    return _handle_cfg(cfg, SupConParams, add_task)


@app.command()
def add_task__supervised_contrast_finetune(
    cfg: Annotated[dict, typer.Option(parser=eval)],
):
    def add_task(builder: "ModelBuilder", cfg):
        return builder.add_task__supervised_contrast_finetune(cfg)

    return _handle_cfg(cfg, SupConParams, add_task)


@app.command()
def checkpoint_evaluations():
    def func(builder: "ModelBuilder"):
        return builder.checkpoint_evaluations()

    return func


@app.command()
def checkpoint_model():
    def func(builder: "ModelBuilder"):
        return builder.checkpoint_model()

    return func


@app.command()
def test():
    def func(builder: "ModelBuilder"):
        builder.dev = False
        return builder

    return func


@app.command()
def run():
    def func(builder: "ModelBuilder"):
        return builder.run()

    return func


@app.command()
def prefix_experiment(prefix: str):
    def func(builder: "ModelBuilder"):
        builder.exp_name = prefix + builder.exp_name
        return builder

    return func


@app.command()
def run_optuna(
    params_cls: Annotated[
        Params,
        typer.Option(
            parser={
                p.__name__: p
                for p in [
                    PatchCoreParams,
                    SupConParams,
                    TSAnomalyParams,
                    TSBackBoneParams,
                ]
            }.__getitem__
        ),
    ],
    optuna_direction: Annotated[OptunaDirection, typer.Option()],
    optuna_metric_name: Annotated[OptunaMetricName, typer.Option()],
    n_trials: Annotated[int, typer.Option()] = 100,
    n_jobs: Annotated[int, typer.Option()] = 1,
    trial_overrides: Annotated[dict, typer.Option(parser=eval)] = "{}",
    timeout: Annotated[int, typer.Option()] = None,
):
    def _run_optuna(model_builder_func: Callable[[Params], "ModelBuilder"]):
        assert isinstance(model_builder_func, Callable)
        from anomaly_detection.model_builder import ModelBuilder

        ModelBuilder.run_optuna(
            build_model_func=model_builder_func,
            params_cls=params_cls,
            optuna_direction=optuna_direction,
            optuna_metric_name=optuna_metric_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            trial_overrides=trial_overrides,
            timeout=timeout,
        )

    return _run_optuna


def _handle_cfg(
    cfg: dict,
    params_cls: typing.Type[Params],
    func: Callable[["ModelBuilder", Params], "ModelBuilder"],
):
    from anomaly_detection.util import load_best_hyperparams

    match cfg:
        case "optuna":  # Optuna will suggest each cfg
            return func
        case (exp_name, direction, metric_name):
            hyperparams = load_best_hyperparams(exp_name, direction, metric_name)
            if "lr_beta1" in hyperparams:
                hyperparams["beta1"] = hyperparams.pop("lr_beta1")
                hyperparams["beta2"] = hyperparams.pop("lr_beta2")
        case dict():
            hyperparams = cfg

    wrapper = functools.partial(func, cfg=params_cls(**hyperparams))
    return functools.update_wrapper(wrapper, wrapped=func)


# Start the CLI
app()
