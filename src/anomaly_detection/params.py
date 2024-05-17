import dataclasses
import itertools
from typing import Annotated

_LR = dict(low=1e-6, high=1, log=True)
_N_SAMPLES = Annotated[
    int, lambda trial: trial.suggest_int("n_samples", 1000, 100_000, log=True)
]

_LAYERS_TO_EXTRACT_FROM = Annotated[
    list[str],
    lambda trial: trial.suggest_categorical(
        "layers_to_extract_from", list(combinations("layer1", "layer2", "layer3"))
    ),
]
_LR_FACTOR_1CYCLE = Annotated[
    float, lambda trial: trial.suggest_float("lr_factor_1cycle", 1, 100, log=True)
]
_LR_LAYER1 = Annotated[float, lambda trial: trial.suggest_float("lr_layer1", **_LR)]
_LR_LAYER2 = Annotated[float, lambda trial: trial.suggest_float("lr_layer2", **_LR)]
_LR_LAYER3 = Annotated[float, lambda trial: trial.suggest_float("lr_layer3", **_LR)]
_LR_LAYER4 = Annotated[float, lambda trial: trial.suggest_float("lr_layer4", **_LR)]
_LR_BOTTLENECK = Annotated[
    float, lambda trial: trial.suggest_float("lr_bottleneck", **_LR)
]
_LR_STUDENT = Annotated[float, lambda trial: trial.suggest_float("lr_student", **_LR)]
_LR_PROJECTOR = Annotated[
    float, lambda trial: trial.suggest_float("lr_projector", **_LR)
]
_BETA1 = Annotated[
    float, lambda trial: trial.suggest_float("beta1", 0.5, 0.999, log=True)
]
_BETA2 = Annotated[
    float, lambda trial: trial.suggest_float("beta2", 0.9, 0.999, log=True)
]
_PIXEL_AGG_FUNC = Annotated[
    str, lambda trial: trial.suggest_categorical("pixel_agg_func", ["mean", "max"])
]


def combinations(*iterable):
    for i in range(1, len(iterable) + 1):
        yield from map(",".join, itertools.combinations(iterable, i))


@dataclasses.dataclass
class SupConParams:
    n_samples: _N_SAMPLES
    lr_layer1: _LR_LAYER1
    lr_layer2: _LR_LAYER2
    lr_layer3: _LR_LAYER3
    lr_layer4: _LR_LAYER4
    lr_projector: _LR_PROJECTOR
    beta1: _BETA1
    beta2: _BETA2
    lr_factor_1cycle: _LR_FACTOR_1CYCLE

    @classmethod
    def best_selfsupervised(cls):
        return cls(
            # Loss: 0.16008920967578888
            beta1=0.18571033545029356,
            beta2=0.9528365694297486,
            lr_layer1=0.00013782784750830402,
            lr_layer2=0.000033872385895682864,
            lr_layer3=0.0000389605380277933,
            lr_layer4=0.00004069973213957303,
            n_samples=52677,
            lr_project=0.000016711519813374378,
        )

    @classmethod
    def best_supervised(cls):
        return cls(
            # Loss: 4.679515838623047
            beta1=0.5332512411569664,
            beta2=0.7169393344334598,
            lr_layer1=0.00006837107196936051,
            lr_layer2=0.00002150535475751489,
            lr_layer3=0.00016868222646047068,
            lr_layer4=0.00012807336914605547,
            lr_project=0.00027440810045856493,
            n_samples=10883,
        )


@dataclasses.dataclass
class PatchCoreParams:
    n_samples: Annotated[int, None]  # Dynamic, cannot exceed train size
    layers_to_extract_from: _LAYERS_TO_EXTRACT_FROM
    pretrain_embed_dimension: Annotated[
        # int, lambda trial: trial.suggest_int("pretrain_embed_dimension", 128, 2048)
        int,
        lambda trial: trial.suggest_int("pretrain_embed_dimension", 32, 512, step=32),
    ]
    target_embed_dimension: Annotated[
        # int, lambda trial: trial.suggest_int("target_embed_dimension", 128, 2048)
        int, lambda trial: trial.suggest_int("target_embed_dimension", 32, 512, step=32)
    ]
    # preprocessing: Annotated[str, lambda trial: trial.suggest_categorical(["mean", "mlp"])]
    # anomaly_scorer_num_nn: Annotated[  # nearest neighbour
    #     int, lambda trial: trial.suggest_int("anomaly_scorer_num_nn", 3, 3)
    # ]
    anomaly_scorer_num_nn: Annotated[  # nearest neighbour
        int, lambda trial: trial.suggest_int("anomaly_scorer_num_nn", 1, 10)
    ]
    patchsize: Annotated[int, lambda trial: trial.suggest_int("patchsize", 1, 50)]
    # patchsize: Annotated[int, lambda trial: trial.suggest_int("patchsize", 3, 3)]
    # patchscore: Annotated[str, lambda trial: trial.suggest_categorical(["max", "mean"])]
    # patchoverlap: Annotated[str, lambda trial: trial.suggest_float(0, 100)]
    feature_retention_pct: Annotated[
        # float, lambda trial: trial.suggest_float("feature_retention_pct", 0.01, 0.2)
        float,
        lambda trial: trial.suggest_float(
            "feature_retention_pct", 0.0001, 0.01, log=True
        ),
    ]
    imagesize: Annotated[int, lambda trial: (3, 224, 224)] = (3, 224, 224)
    pixel_agg_func: Annotated[str, lambda trial: "max"] = "max"


@dataclasses.dataclass
class TSBackBoneParams:
    n_samples: _N_SAMPLES
    lr_layer1: _LR_LAYER1
    lr_layer2: _LR_LAYER2
    lr_layer3: _LR_LAYER3
    lr_bottleneck: _LR_BOTTLENECK
    lr_student: _LR_STUDENT
    beta1: _BETA1
    beta2: _BETA2
    lr_factor_1cycle: _LR_FACTOR_1CYCLE


@dataclasses.dataclass
class TSAnomalyParams:
    n_samples: _N_SAMPLES
    layers_to_extract_from: _LAYERS_TO_EXTRACT_FROM
    lr_bottleneck: _LR_BOTTLENECK
    lr_student: _LR_STUDENT
    beta1: _BETA1
    beta2: _BETA2
    lr_factor_1cycle: _LR_FACTOR_1CYCLE
    pixel_agg_func: _PIXEL_AGG_FUNC


class Params:
    __mapping__ = dict(
        PatchCoreParams=PatchCoreParams,
        SupConParams=SupConParams,
        TSAnomalyParams=TSAnomalyParams,
        TSBackBoneParams=TSBackBoneParams,
    )

    @classmethod
    def from_str(cls, s: str):
        if s not in cls.__mapping__:
            raise KeyError(
                f"Invalid param name: {s}. Valid: {list(Params.__mapping__)}"
            )
        return cls.__mapping__[s]
