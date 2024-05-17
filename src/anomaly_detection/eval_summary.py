import functools
import typing

import hvplot.pandas  # noqa
import pandas as pd
import panel as pn

from anomaly_detection import config

pn.extension()

common_kwargs = dict(
    height=500,
    width=500,
    rot=90,
    grid=True,
    shared_axes=False,
    ylim=(0, 100),
    xlabel="BB dataset, BB method+Anomaly method",
    ylabel="AUC",
    yformatter="%.0f%%",
    fontsize={
        "title": 18,
        "labels": 16,
        "xticks": 16,
        "yticks": 16,
    },
)


def dropna(plot, element, valid):
    plot.state.x_range.factors = [
        x for x in plot.state.x_range.factors if tuple(x) in valid
    ]


def anomaly_scavport_auc(
    df_summary: pd.DataFrame, y: typing.Literal["anomaly_auc@OPT", "anomaly_auc@TEST"]
):
    df = df_summary.query('task.eq("anomaly") and clf_ds.eq("scavport")').copy()
    df[y] *= 100
    plot = (
        df.hvplot.bar(
            y=y,
            x="bb_ds",
            by="methods",
            hover_cols=config.EVAL_GROUPING,
            **common_kwargs,
        ).opts(
            title="Scavenge port Anomaly Detection",
            hooks=[
                functools.partial(
                    dropna, valid=set(map(tuple, df[["bb_ds", "methods"]].to_numpy()))
                )
            ],
        )
    ).sort()
    return plot
    # df_summary.query('task.eq("anomaly") and clf_ds.eq("scavport")').hvplot.scatter(
    #     y="anomaly_auc@OPT",
    #     by="bb_ds",
    #     x="methods",
    #     rot=30,
    #     height=600,
    #     width=900,
    #     hover_cols=grouping,
    #     alpha=0.75,
    # ).opts(
    #     title="Anomaly Detection on Scavport dataset (Best Optuna)",
    #     fontsize={
    #         "title": 14,
    #         "labels": 14,
    #         "xticks": 14,
    #         "yticks": 14,
    #     },
    # )


def anomaly_condition_auc(
    df_summary: pd.DataFrame, y: typing.Literal["anomaly_auc@OPT", "anomaly_auc@TEST"]
):
    df = df_summary.query(
        'task.eq("anomaly") and clf_ds.str.contains("condition")'
    ).copy()
    df["clf_ds"] = df.clf_ds.str.replace("^condition-", "", regex=True)
    df[y] *= 100
    condition_names = df.clf_ds.unique()
    agg = df.groupby(["bb_ds", "methods"]).agg(
        {y: ["max", "mean", "min"], "clf_ds": "unique"}
    )
    opts_kwargs = dict(
        hooks=[
            functools.partial(
                dropna, valid=set(map(tuple, df[["bb_ds", "methods"]].to_numpy()))
            )
        ]
    )
    kwargs = dict(
        x="bb_ds",
        by="methods",
        **common_kwargs,
    )

    agg_plot_simple = (
        agg.xs(y, axis=1, level=0)
        .hvplot.bar(
            title="Mean Condition Anomaly Detection",
            y="mean",
            **kwargs,
        )
        .sort()
    ).opts(**opts_kwargs)
    agg_plot_minmax = (
        agg.xs(y, axis=1, level=0)
        .stack()
        .hvplot.bar(
            title="{{min,mean,max}} Condition Anomaly Detection",
            **kwargs,
            alpha=0.50,
        )
        .sort()
    ).opts(**opts_kwargs)
    plots = df.hvplot.bar(
        y=y,
        groupby="clf_ds",
        **kwargs,
    ).opts(title="Anomaly Detection on Condition datasets", **opts_kwargs)
    individ_plot = [
        plots[clf_ds].sort().opts(title=clf_ds) for clf_ds in condition_names
    ]
    agg_plot_simple
    return pn.Column(agg_plot_simple, agg_plot_minmax, pn.Row(*individ_plot))
