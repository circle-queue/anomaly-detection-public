import functools
import itertools
import pickle
import typing

import mlflow
import mlflow.experiments
import numpy as np
import pandas as pd
import panel as pn

from anomaly_detection import config

pn.extension()


@functools.cache
def load_runs_df():
    cache_path = config.DATA_ROOT / "mlruns.pkl"
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))
    mlflow.set_tracking_uri(config.REPO_DATA_ROOT.parents[2] / "mlruns")
    all_runs = mlflow.search_runs(search_all_experiments=True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    exp_id_to_name = {e.experiment_id: e.name for e in mlflow.search_experiments()}
    all_runs = all_runs.assign(exp_name=lambda df: df.experiment_id.map(exp_id_to_name))
    cache_path.write_bytes(pickle.dumps(all_runs))
    return all_runs


def exp_name_to_run_info(df: pd.DataFrame) -> pd.DataFrame:
    df[["task", "clf_ds", "model", "bb_model"]] = df.exp_name.str.split(
        "__", expand=True
    )
    df[["bb_model", "patchcore"]] = (
        df["bb_model"].str.extract(r"^(.+?)(_patchcore)?$").fillna("")
    )
    df["model"] = np.where(df["patchcore"], "patchcore", df["model"])
    df[["optuna", "task"]] = df["task"].str.extract(r"(optuna_)?(.*)")
    df[["test", "task"]] = df["task"].str.extract(r"(test_)?(.*)")
    df["optuna"] = df.optuna.fillna("").str.contains("optuna")
    df["test"] = df.test.fillna("").str.contains("test")

    df["url"] = df.apply(
        lambda x: f"{mlflow.get_tracking_uri()}/#experiments/{x.experiment_id}/runs/{x.run_id}",
        axis=1,
    )
    df[["bb_method", "bb_ds"]] = (
        df.reset_index()
        .bb_model.str.replace("resnet18_", "")
        .str.extract(r"^(.+?)@(.*)$")
        .fillna("imagenet")
        .squeeze()
        .values
    )
    df["methods"] = (
        (df.bb_method + "+" + df.model)
        .str.replace("resnet18_?", "", regex=True)
        .str.lower()
    )
    df["bb_ds_methods"] = (df.bb_ds + "+" + df.methods).str.replace(
        "resnet18_?", "", regex=True
    )
    df["bb_ds+clf_method"] = df["bb_ds"] + "+" + df["clf_ds"]
    df[config.EVAL_GROUPING]
    df[config.EVAL_DERIVED]
    return df


def one_or_none(df, exp_name) -> None | str:
    if len(df) == 0:
        return None
    elif len(df) > 1:
        raise ValueError(f"Multiple runs found for {exp_name = }")
    return df.iloc[0]


@functools.cache
def load_summary_df():
    # cache_path = config.DATA_ROOT / "df_summary.pkl"
    # if cache_path.exists():
    #     return pickle.load(open(cache_path, "rb"))

    maximize = "metrics.anomaly_auc"
    minimize = "metrics.eval_loss"

    all_runs = load_runs_df()
    all_runs = exp_name_to_run_info(all_runs)

    exp_id_to_optuna_runs = {}
    # for exp_id, df in all_runs.groupby("experiment_id"):
    #     for run_id in df[df["tags.mlflow.parentRunId"].isnull()].run_id:
    #         artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    #         for artifact in artifacts:
    #             if Path(artifact.path).suffix == ".sqlite3":
    #                 break
    #         else:
    #             print(f"No Optuna DB found for {exp_id = }")
    #             continue

    #         artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    #         with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
    #             dst = mlflow.artifacts.download_artifacts(
    #                 dst_path=temp_dir, artifact_uri=f"{artifact_uri}/{artifact.path}"
    #             )
    #             study = optuna.load_study(None, f"sqlite:///{temp_dir}/{artifact.path}")
    #             exp_id_to_optuna_runs[exp_id] = len(study.trials)

    done_runs = all_runs.query(
        'status == "FINISHED" and exp_name != "Default" and not exp_name.str.contains("ablation")'
    )[lambda df: df[maximize].notnull() | df[minimize].notnull()].copy()

    summary = []
    for group, df in done_runs.groupby(config.EVAL_GROUPING):
        exp_name = df.exp_name.iloc[0]
        assert all("optuna" not in name for name in group)

        optuna_df = df.query("optuna")
        best_df = one_or_none(df.query("not optuna and not test"), exp_name)
        test_df = one_or_none(df.query("test"), exp_name)

        run_number = optuna_df["tags.mlflow.runName"].astype(int)
        if not run_number.is_monotonic_decreasing:
            raise ValueError(
                f"Optuna runs not performed in order: {group = } {exp_name = }"
            )

        metric_name = maximize if df[maximize].notnull().any() else minimize
        optuna_idx = (
            df[maximize] if metric_name == maximize else -df[minimize]
        ).argmax()

        to_track = [metric_name, "url"]
        best_optuna_data = {
            f"{k}@OPT": v for k, v in df.iloc[optuna_idx].items() if k in to_track
        }
        if best_df is not None:
            best_run_data = {f"{k}@DEV": v for k, v in best_df.items() if k in to_track}
        else:
            best_run_data = {f"{k}@DEV": None for k in to_track}
        if test_df is not None:
            test_run_data = {
                f"{k}@TEST": v for k, v in test_df.items() if k in to_track
            }
        else:
            test_run_data = {f"{k}@TEST": None for k in to_track}

        n_optuna_runs = df.experiment_id.map(exp_id_to_optuna_runs).max()
        assert (
            df[config.EVAL_DERIVED].nunique().eq(1).all()
        ), f"Derived metadata columns must be unique: {df[config.EVAL_DERIVED].nunique() = }"
        data = (
            dict(zip(config.EVAL_GROUPING, group))
            | {k: df[k].iloc[0] for k in config.EVAL_DERIVED}
            | best_optuna_data
            | best_run_data
            | test_run_data
            | {"n_optuna_runs": len(optuna_df)}
            | {"n_optuna_runs_true": n_optuna_runs}
        )
        summary.append(data)

    tasks = ["anomaly", "embed_loss"]
    clf_dss = [
        "condition-images-scavengeport-overview",
        "condition-lock-condition",
        "condition-ring-condition",
        "condition-ring-surface-condition",
        "scavport",
        "vesselarchive",
    ]
    models = ["patchcore", "resnet18_T-S", "resnet18_self-sup-con", "resnet18_sup-con"]
    bb_models = [
        "imagenet",
        "resnet18_T-S@scavport",
        "resnet18_T-S@vesselarchive",
        "resnet18_self-sup-con@scavport",
        "resnet18_self-sup-con@vesselarchive",
        "resnet18_sup-con@scavport",
    ]

    class Index(typing.NamedTuple):
        task: str
        clf_ds: str
        model: str
        bb_model: str

    expected_idx = {
        Index(task, clf_ds, model, bb_model)
        for task, clf_ds, model, bb_model in itertools.product(
            tasks, clf_dss, models, bb_models
        )
    }

    def kill(idx):
        assert idx not in df.index
        expected_idx.remove(idx)

    for idx in list(expected_idx):
        if "vesselarchive" in idx.clf_ds and any("resnet18_sup-con" in i for i in idx):
            kill(idx)  # Impossible, no labels
        elif "anomaly" in idx and "vesselarchive" in idx.clf_ds:
            kill(idx)  # Impossible, vesselarchive can only fine-tune
        elif "embed_loss" in idx and any("@" in i for i in idx):
            kill(idx)  # Impossible, fine-tuning a fine-tuned model
        elif "embed_loss" in idx and "patchcore" in idx:
            kill(idx)  # Impossible, patchcore cannot fine-tune embeddings
        elif "embed_loss" in idx and "condition" in idx.clf_ds:
            kill(idx)  # Impossible, condition is for anomaly detection
        elif "anomaly" in idx and "sup-con" in idx.model:
            kill(idx)  # Impossible, sup-con is for embedding loss

    df_summary = (
        pd.DataFrame(summary)
        .set_index(config.EVAL_GROUPING)
        .reindex(list(map(tuple, expected_idx)))
        .sort_index(axis="index")
        .sort_index(axis="columns")
        .fillna({"n_optuna_runs": 0})
        # .loc[pd.IndexSlice["anomaly", :, "resnet18_T-S", :]]
        .reset_index()
    )
    df_summary.columns = df_summary.columns.str.replace("metrics.", "")
    # cache_path.write_bytes(pickle.dumps(df_summary))
    return df_summary


def formatted_summary_df(df_summary: pd.DataFrame):
    df = df_summary.copy()
    df["delta"] = df_summary["anomaly_auc@OPT"] - df_summary["anomaly_auc@DEV"]
    return df.set_index(config.EVAL_GROUPING).style.format(
        {
            "delta": lambda x: f'<span style="color: {"red" if x > 0.01 else "black"}">{x:.2f}</span>',
            "url@DEV": lambda x: f'<a href="{x}">DEV</a>',
            "url@OPT": lambda x: f'<a href="{x}">OPT</a>',
            "url@TEST": lambda x: f'<a href="{x}">TEST</a>',
            "n_optuna_runs": lambda x: f'<span style="color: {"red" if x < 100 else "black"}">{x}</span>',
        },
        precision=2,
        na_rep="",
    )


if __name__ == "__main__":
    df_summary = load_summary_df()
    formatted_df = formatted_summary_df(df_summary)
    pn.serve(formatted_df)
