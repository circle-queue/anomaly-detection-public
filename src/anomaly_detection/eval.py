# panel serve --autoreload src/anomaly_detection/eval.py
import datetime as dt
import functools
import itertools

import bokeh.io
import bokeh.palettes
import cv2
import holoviews as hv
import holoviews.plotting.links
import holoviews.streams
import hvplot.pandas  # noqa: F401 Adds .hvplot to pandas dataframes
import mlflow
import numpy as np
import pandas as pd
import panel as pn
import rd4ad._modules
import rd4ad.resnet_models
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from bokeh.models import NumeralTickFormatter
from PIL import Image
from sklearn import manifold, metrics
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from anomaly_detection import (
    config,
    eval_summary,
    mlflow_summary,
    model_builder,
    models,
    util,
)

fontsize = {
    "title": 18,
    "labels": 16,
    "xticks": 16,
    "yticks": 16,
}


def pct_fmt():
    """Must be function to avoid NumeralTickFormatter(id=...) is already in a doc"""
    return NumeralTickFormatter(format="0%")


DOCS_PATH = config.REPO_DATA_ROOT.parents[2] / "docs"


def exception_handler(ex):
    pn.state.notifications.error(f"Error: {ex}")


pn.extension("tabulator", exception_handler=exception_handler, notifications=True)

pn.param.ParamMethod.loading_indicator = True

PLOT_KWARGS = {"width": 1000}

ablation_patchsize_exp_name = (
    "ablation-patchsize_anomaly__scavport__resnet18__imagenet_patchcore"
)

name_to_experiment = {
    e.name: e for e in mlflow.search_experiments() if "optuna_" not in e.name
}


def main_anomaly():
    """
    >>> exp_name = 'anomaly__scavport__resnet18_T-S__imagenet'
    >>> exp_name = 'anomaly__scavport__resnet18__imagenet_patchcore'
    >>> exp_name = 'anomaly__condition-ring-condition__resnet18_T-S__imagenet'
    """
    experiment_widget = pn.widgets.AutocompleteInput(
        value="anomaly__scavport__resnet18_T-S__imagenet",
        options=sorted(name_to_experiment),
        name="experiment",
        search_strategy="includes",
        width=1000,
    )

    @pn.depends(experiment_widget)
    def _inner(exp_name):
        dashboard = pn.Column(
            experiment_widget,
            f"# Dev dataset {exp_name = }",
        )
        if "anomaly" not in exp_name:
            dashboard.append(pn.Column("### Select an anomaly detection experiment"))
            return dashboard

        dloads = get_dataloaders(exp_name)
        dashboard.append(
            get_class_frequency_plot(dloads.train).opts(title="Train class frequency")
        )
        dashboard.append(
            get_class_frequency_plot(dloads.test).opts(title="Dev class frequency")
        )

        df = util.load_checkpoint_df(exp_name, dloads.test, train_pcts=(100,))
        metric_keys = sorted(df.metrics.iloc[0])

        anomaly_tag_w = pn.widgets.Select(options=metric_keys)

        @pn.depends(anomaly_tag_w)
        def _plot(anomaly_tag):
            output = pn.Column(*dashboard)
            output.append(roc_curve(df, dloads, y_score_col=anomaly_tag))

            if "patchcore" not in exp_name:
                output.append(
                    anomaly_train_progress_plot(exp_name, dloads, anomaly_tag)
                )

            img_panel = img_table(df, metric_keys, get_expand_content_func(exp_name))

            return pn.Column(anomaly_tag_w, output, img_panel)

        return pn.Column(_plot)

    return pn.Column(_inner)


def get_expand_content_func(exp_name, run_id=None):
    if not run_id:
        run_id = util.run_id_from_exp_name(exp_name)  # Should have exactly one run
    dloads = get_dataloaders(exp_name)
    transform = models.transform_dict["resnet18_crop_only"]
    out_size = (512, 512)

    def expand_content(row):
        """
        >>> row = next(df.itertuples())
        """
        img_raw = Image.open(row.path)
        assert "resnet18" in exp_name
        processed_img = dloads.test.dataset.transform(img_raw).to("cuda").unsqueeze(0)
        img: Image = transform(img_raw)
        # Load here to only crash if we expand the content and faiss-gpu is not available
        model = util.load_model_cached(run_id, train_pct=100).to("cuda")

        with torch.no_grad():
            if "patchcore" not in exp_name:
                layers = mlflow.get_run(run_id).data.params["layers_to_extract_from"]
                input_layers, output_layers = util.filter_layers(
                    *model(processed_img.to(config.DEVICE)),
                    keep_layers=layers.split(","),
                )
                anomaly_map = util.gaussian(
                    util.upscaled_anomaly_maps(
                        input_layers, output_layers, out_size=img.size[::-1]
                    )
                )
                anomaly_map = anomaly_map.cpu().numpy()
            else:
                from patchcore import patchcore

                assert isinstance(model, patchcore.PatchCore)
                [_pred], [anomaly_map], *_ = model.predict(processed_img)

        grayscale = np.uint8((minmax_scale(anomaly_map) * 254).squeeze())
        heatmap = cv2.applyColorMap(255 - grayscale, cv2.COLORMAP_JET)
        merged = Image.fromarray(
            (np.float32(img) / 2 + np.float32(heatmap) / 2)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return pn.Swipe(img.resize(out_size), merged.resize(out_size), value=1)

    return expand_content


def anomaly_train_progress_plot(
    exp_name, dloads: util.TrainTestDataLoaders, anomaly_tag: str
):
    df = util.load_checkpoint_df(exp_name, dloads.test)
    df[["path", "target"]] = dloads.test.dataset.imgs * len(config.CKP_PCTS)
    assert len(df[["class_label", "target"]].value_counts()) == df.class_label.nunique()
    df["is_outlier"] = dloads.test.dataset.is_outlier * len(config.CKP_PCTS)
    cmap = dict.fromkeys(df[df.is_outlier].class_label.unique(), "red") | dict.fromkeys(
        df[~df.is_outlier].class_label.unique(), "green"
    )

    kwargs = {
        "color": "class_label",
        "cmap": cmap,
        "legend": False,
        "rot": 90,
        "width": 1000,
    }
    recall_w = pn.widgets.FloatSlider(start=0, end=1, value=0.50, name="Recall")

    @pn.depends(recall_w)
    def anomaly_score_plots(recall: float):
        loss_timeseries_plot = df.hvplot.box(
            anomaly_tag, by=["class_label", "train_pct"], **kwargs
        ).opts(
            title=f"""{anomaly_tag!r} per class over time when trained on all classes.
        🟩⬇Lower is better. 🟥⬆Higher is better."""
        )

        thresh = df[df.is_outlier].groupby("train_pct")[anomaly_tag].quantile(recall)

        outlier_score = (
            df.groupby("train_pct")[anomaly_tag]
            .transform(lambda s: s - thresh[s.name])
            .rename("outlier_score")
            .groupby([df.train_pct, df.class_label])
            .apply(lambda s: s)
            .reset_index()
        )
        relative_outlier_plot = outlier_score.hvplot.box(
            "outlier_score", ["class_label", "train_pct"], grid=True, **kwargs
        ).opts(
            title=f"""Relative outlier score per class over time given {recall = } wrt. outlier devset
        🟩⬇Lower is better. 🟥⬆Higher is better."""
        )

        avg_1class_ratio = (
            outlier_score.groupby(["train_pct", "class_label"])
            .agg(
                pred_1class_ratio=pd.NamedAgg(
                    "outlier_score", lambda s: (s < 0).mean() * 100
                )
            )
            .reset_index()
        )
        pred_ratio_plot = avg_1class_ratio.hvplot.bar(
            x="class_label",
            y="pred_1class_ratio",
            by="train_pct",
            **kwargs,
            ylabel="P(y=one_class)",
            yformatter="%.0f%%",
        ) * hv.HLine(recall * 100).opts(color="black", line_dash="dotted").opts(
            title="""% of '1class' predictions by class.
        🟩⬆Higher is better. 🟥⬇Lower is better."""
        )
        return pn.Column(loss_timeseries_plot, relative_outlier_plot, pred_ratio_plot)

    return pn.Column(recall_w, anomaly_score_plots)


def roc_curve(df, dloads: util.TrainTestDataLoaders, y_score_col: str):
    roc_kwargs = dict(y_true=dloads.test.dataset.is_outlier, y_score=df[y_score_col])
    auc_score = metrics.roc_auc_score(**roc_kwargs)
    fpr, tpr, thresh = metrics.roc_curve(**roc_kwargs)
    return pd.DataFrame(
        {"fpr": fpr * 100, "tpr": tpr * 100, "thresh": thresh}
    ).hvplot.line(
        x="fpr",
        y="tpr",
        title=f"ROC curve [AUC: {auc_score:.2%}]",
        hover_cols=["thresh"],
        yformatter="%.0f%%",
        xformatter="%.0f%%",
        aspect=1,
    )


def main_embed():
    experiments = ["imagenet"] + [
        name for name in name_to_experiment if "embed" in name
    ]

    @functools.cache
    def get_embed_df(exp_name: str, eval_dataset: list[str]):
        test_loader = get_dataloaders(eval_dataset).test
        model = (
            (
                util.load_model_cached(exp_name, train_pct=100)
                if exp_name != "imagenet"
                else rd4ad.resnet_models.resnet18()
            )
            .to(config.DEVICE)
            .eval()
            .requires_grad_(False)
        )
        idx_to_class = util.idx_to_class(test_loader)
        embeds = []
        classes = []
        paths = []
        for (img, class_), path in util.with_paths(test_loader):
            img = img.to(config.DEVICE)
            if isinstance(model, models.SupConResNet):
                model = supcon_to_rd4ad_encoder(model).to(config.DEVICE)
            elif isinstance(model, models.EncoderBottleneckDecoderResNet):
                model = model.teacher_encoder
            teacher_layers = model(img)
            embed = torch.nn.AdaptiveAvgPool2d((1, 1))(teacher_layers[-1]).squeeze()
            classes += [idx_to_class[c] for c in class_.tolist()]
            embeds.extend(embed.cpu().numpy())
            paths.extend(path)
        return pd.DataFrame(
            [classes, embeds, paths], index=["class_label", "embed", "path"]
        ).T.assign(train_pct=100)

    datasets = [
        "embed_scavport",
        "embed_vesselarchive",
        "anomaly_scavport",
    ]
    embed_df = pd.concat(
        [
            get_embed_df(exp_name, ds).assign(eval_dataset=ds, exp_name=exp_name)
            for ds, exp_name in tqdm(list(itertools.product(datasets, experiments)))
        ],
        ignore_index=True,
    ).query('class_label != "inlier"')  # Duplicated with embed_scavport
    embed_df["class_label"] = embed_df["class_label"].replace({"dev": "Vessel archive"})
    cmap = {
        "Vessel archive": "dimgrey",
        "outlier": "red",
        "liner": bokeh.palettes.Light8[0],
        "topland": bokeh.palettes.Light8[1],
        "single-piston-ring": bokeh.palettes.Light8[2],
        "skirt": bokeh.palettes.Light8[3],
        "piston-ring-overview": bokeh.palettes.Light8[4],
        "piston-top": bokeh.palettes.Light8[5],
        "scavange-box": bokeh.palettes.Light8[6],
        "piston-rod": bokeh.palettes.Light8[7],
    }

    @functools.cache
    def tsne_cols(group, perplexity):
        # This func just supports fast debugging
        # We just use group as an identifier for caching
        exp_df[["dim1", "dim2"]] = manifold.TSNE(
            n_components=2, n_jobs=8, perplexity=perplexity
        ).fit_transform(np.stack(exp_df["embed"]))
        return exp_df

    tsne_dfs = []
    for group, exp_df in tqdm(
        list(embed_df.groupby("exp_name")), desc="Computing TSNE"
    ):
        tsne_dfs.append(tsne_cols(group, perplexity=100))
    tsne_df = pd.concat(tsne_dfs, ignore_index=True).sort_values(
        "class_label",
        key=lambda x: x.map({"Vessel archive": 0, "outlier": 2}).fillna(1),
    )
    tsne_df["size"] = tsne_df.class_label.map({"Vessel archive": 1}).fillna(10)

    plot_kwargs = dict(
        x="dim1",
        y="dim2",
        groupby="exp_name",
        size="size",
        color="class_label",
        cmap=cmap,
        hover_cols="path",
        xaxis=None,
        yaxis=None,
        width=300,
        height=300,
        padding=0,
        tools=["hover", "tap", "lasso_select"],
        nonselection_alpha=0.02,
    )
    points = tsne_df.hvplot.points(**plot_kwargs).options(show_legend=False)

    def image_on_click():
        imgs = pn.GridBox(ncols=2)
        self_sup_plot = points[
            "embed_loss__vesselarchive__resnet18_self-sup-con__imagenet"
        ].opts(title='Self-sup-con. "Vessel archive"')
        baseline_plot = points["imagenet"].options(
            tools=[], toolbar=None, title="ImageNet"
        )
        layout = pn.Row(baseline_plot, self_sup_plot, imgs)

        last = dt.datetime.now()

        def update_img(index):
            nonlocal last
            if last + dt.timedelta(seconds=1) > dt.datetime.now():
                return
            last = dt.datetime.now()

            selected_imgs = self_sup_plot.dframe().iloc[index].path.tolist()
            layout[0] = (
                baseline_plot.select(path=selected_imgs) if index else baseline_plot
            )
            # imgs[:] = list(map(Image.open, selected_imgs[:4]))

        _selection = holoviews.streams.Selection1D(
            source=self_sup_plot, subscribers=[update_img]
        )

        return layout

    def all_datasets_plot():
        plot_order = [
            # T-S
            "imagenet",
            "embed_loss__scavport__resnet18_T-S__imagenet",
            "embed_loss__vesselarchive__resnet18_T-S__imagenet",
            # sup-con
            "embed_loss__scavport__resnet18_sup-con__imagenet",
            # self-sup-con
            "embed_loss__scavport__resnet18_self-sup-con__imagenet",
            "embed_loss__vesselarchive__resnet18_self-sup-con__imagenet",
        ]
        ds_plots = [
            points[plot_name].opts(
                toolbar=None,
                hooks=[remove_bokeh_logo],
                title=plot_name.replace("__imagenet", "")
                .replace("resnet18_", "")
                .replace("embed_loss__", "")
                .replace("__", " ")
                .capitalize(),
                show_legend=False,
            )
            for plot_name in plot_order
        ]
        return hv.Layout(ds_plots).cols(3).opts(toolbar=None)

    def scavport_class_and_anomaly():
        return (
            tsne_df.query('eval_dataset.isin(["embed_scavport", "anomaly_scavport"])')
            .hvplot.scatter(**plot_kwargs)
            .opts(show_legend=False, tools=["tap"])
        ).opts(title='Scavport "embed" and "anomaly" datasets')

    _all_datasets_plot = all_datasets_plot()
    # hv.save(_all_datasets_plot, DOCS_PATH / "all_datasets_embed_tsne_plot_ppx_20.png")

    return pn.Column(
        image_on_click(),
        _all_datasets_plot,
        scavport_class_and_anomaly(),
    )


def sam_embed():
    img_size = 224
    model_names = [
        "sam_vit_b_01ec64.pth",
        "sam_vit_l_0b3195.pth",
        "sam_vit_h_4b8939.pth",
    ]
    img_sizes = [224, 512, 1024]
    embed_dfs = {}
    for model_name, img_size in itertools.product(model_names, img_sizes):
        exp_name = f'{model_name.rpartition("_")[0]}_{img_size}'
        encoder = (
            models.SamMaeEncoder(model_name, img_size)
            .to(config.DEVICE)
            .eval()
            .requires_grad_(False)
        )

        dl_kwargs = {
            "batch_size": 1,
            "transform": models.transform_dict[f"sam{img_size}"],
        }

        @functools.cache
        def get_embed_df(eval_dataset: str):
            test_loader = get_dataloaders(eval_dataset, dl_kwargs).test
            idx_to_class = util.idx_to_class(test_loader)
            embeds = []
            classes = []
            paths = []
            for (img, class_), path in tqdm(
                util.with_paths(test_loader),
                desc=eval_dataset,
                leave=False,
                total=len(test_loader),
            ):
                img = img.to(config.DEVICE)
                embed = encoder(img)
                classes += [idx_to_class[c] for c in class_.tolist()]
                embeds.extend(embed.cpu().numpy())
                paths.extend(path)
            return pd.DataFrame(
                [classes, embeds, paths], index=["class_label", "embed", "path"]
            ).T.assign(train_pct=100, exp_name=exp_name)

        datasets = [
            "embed_scavport",
            "embed_vesselarchive",
            "anomaly_scavport",
        ]
        embed_df = pd.concat(
            [get_embed_df(ds).assign() for ds in tqdm(datasets)],
            ignore_index=True,
        ).query('class_label != "inlier"')  # Duplicated with embed_scavport
        embed_dfs[exp_name] = embed_df

    embed_df["class_label"] = embed_df["class_label"].replace({"dev": "Vessel archive"})
    cmap = {
        "Vessel archive": "dimgrey",
        "outlier": "red",
        "liner": bokeh.palettes.Light8[0],
        "topland": bokeh.palettes.Light8[1],
        "single-piston-ring": bokeh.palettes.Light8[2],
        "skirt": bokeh.palettes.Light8[3],
        "piston-ring-overview": bokeh.palettes.Light8[4],
        "piston-top": bokeh.palettes.Light8[5],
        "scavange-box": bokeh.palettes.Light8[6],
        "piston-rod": bokeh.palettes.Light8[7],
    }

    @functools.cache
    def tsne_cols(group, perplexity):
        # This func just supports fast debugging
        # We just use group as an identifier for caching
        exp_df[["dim1", "dim2"]] = manifold.TSNE(
            n_components=2, n_jobs=8, perplexity=perplexity
        ).fit_transform(np.stack(exp_df["embed"]))
        return exp_df

    tsne_dfs = []
    for group, exp_df in tqdm(
        list(embed_df.groupby("exp_name")), desc="Computing TSNE"
    ):
        tsne_dfs.append(tsne_cols(group, perplexity=100))
    tsne_df = pd.concat(tsne_dfs, ignore_index=True).sort_values(
        "class_label",
        key=lambda x: x.map({"Vessel archive": 0, "outlier": 2}).fillna(1),
    )
    tsne_df["size"] = tsne_df.class_label.map({"Vessel archive": 1}).fillna(10)

    plot_kwargs = dict(
        x="dim1",
        y="dim2",
        groupby="exp_name",
        size="size",
        color="class_label",
        cmap=cmap,
        hover_cols="path",
        xaxis=None,
        yaxis=None,
        width=300,
        height=300,
        padding=0,
        tools=["hover", "tap", "lasso_select"],
        nonselection_alpha=0.02,
    )
    points = tsne_df.hvplot.points(**plot_kwargs).options(show_legend=False)
    pn.serve(points)


def tsne_plot(df, embed_dim="embed", **plot_kwargs):
    df[["dim1", "dim2"]] = manifold.TSNE(n_components=2).fit_transform(
        np.stack(df[embed_dim])
    )

    train_pct = pn.widgets.Select(
        value=100, options=[*set(df.train_pct)], name="train_pct"
    )

    @pn.depends(train_pct)
    def plot(train_pct):
        return df[df.train_pct == train_pct].hvplot.scatter(
            "dim1", "dim2", color="class_label", **(PLOT_KWARGS | plot_kwargs)
        )

    return pn.Column("### TSNE of embeddings", train_pct, plot)


def img_table(
    df,
    metrics: list[str] = ["loss"],
    row_content=lambda row: Image.open(row.path),
):
    cols = ["train_pct", "class_label", *metrics, "path"]
    img_table = pn.widgets.Tabulator(
        df[cols],
        row_content=row_content,
        disabled=True,
        hidden_columns=["index"],
        header_filters=True,
    )

    pickers = pn.Row()
    for col in ["train_pct", "class_label"]:
        picker = pn.widgets.MultiSelect(
            options=[*sorted(set(df[col]))], size=10, name=col
        )
        img_table.add_filter(picker, col)
        pickers.append(picker)
    pickers[0].value = [100]
    return pn.Column("### Examine images", pickers, img_table)


def get_dataloaders(
    experiment_name,
    dl_kwargs={"batch_size": 8, "transform": models.transform_dict["resnet18"]},
):
    builder = model_builder.ModelBuilder()
    builder.dev = "test" not in experiment_name

    if "condition" in experiment_name:
        _, ds, *_ = experiment_name.split("__")
        _, _, sub_dataset = ds.partition("-")
        test_classes = [
            p.name for p in (config.CONDITION_ROOT / sub_dataset / "dev").iterdir()
        ]
        builder.add_condition_dataset(
            sub_dataset=sub_dataset,
            train_classes=["good"],
            test_classes=test_classes,
            **dl_kwargs,
        )
        builder.set_outliers(
            labels_inliers=["good"],
            labels_outlier=[c for c in test_classes if c != "good"],
        )
    elif "scavport" in experiment_name and "embed" in experiment_name:
        builder.add_scavport_dataset(
            train_classes=config.SCAVPORT_CLF_CLASSES,
            test_classes=config.SCAVPORT_CLF_CLASSES,
            **dl_kwargs,
        )
    elif "scavport" in experiment_name and "anomaly" in experiment_name:
        builder.add_scavport_dataset(
            train_classes=config.SCAVPORT_CLF_CLASSES,
            test_classes=["inlier", "outlier"],
            **dl_kwargs,
        )
        builder.set_outliers(labels_inliers=["inlier"], labels_outlier=["outlier"])
    elif "vessel" in experiment_name:
        builder.add_vessel_archive_dataset(**dl_kwargs)
    else:
        raise NotImplementedError(experiment_name)
    return util.TrainTestDataLoaders(builder.train_dataloader, builder.test_dataloader)


def get_class_frequency_plot(test_dataloader: DataLoader):
    return (
        pd.Series(test_dataloader.dataset.targets)
        .map(util.idx_to_class(test_dataloader))
        .value_counts()
        .loc[test_dataloader.dataset.classes]
        .hvplot.bar(height=150, title="#Samples")
    )


def main_summary():
    df_summary = mlflow_summary.load_summary_df()
    latex = (
        df_summary[
            [
                "clf_ds",
                "model",
                "bb_model",
                "bb_ds",
                "anomaly_auc@OPT",
                "anomaly_auc@TEST",
            ]
        ]
        .pipe(lambda df: df.set_index(df.columns.tolist()).sort_index().reset_index())
        .dropna()
        .to_latex(index=False, float_format="%.2f")
        .replace("_", "-")
        .replace("condition-", "")
        .replace("images-", "")
        .replace("resnet18-", "")
        .replace("anomaly-", "")
    )
    print(latex)

    scavport_auc, condition_auc = [
        pn.Swipe(
            value=5,
            *[func(df_summary, x) for x in ["anomaly_auc@OPT", "anomaly_auc@TEST"]],
        )
        for func in [
            eval_summary.anomaly_scavport_auc,
            eval_summary.anomaly_condition_auc,
        ]
    ]
    # hv.save(
    #     eval_summary.anomaly_scavport_auc(df_summary, "anomaly_auc@TEST").opts(
    #         toolbar=None
    #     ),
    #     DOCS_PATH / "scavport_auc.png",
    # )
    # hv.save(
    #     hv.Layout(
    #         [
    #             subset_plot.object.opts(toolbar=None)
    #             for subset_plot in eval_summary.anomaly_condition_auc(
    #                 df_summary, "anomaly_auc@TEST"
    #             )[-1]
    #         ],
    #     ).cols(2),
    #     DOCS_PATH / "condition_auc.png",
    # )

    return pn.Column(
        scavport_auc,
        condition_auc,
        mlflow_summary.formatted_summary_df(df_summary),
    )


def patchcore_patchsize_auc():
    runs_df = mlflow.search_runs(experiment_names=[ablation_patchsize_exp_name])
    runs_df["patchsize"] = runs_df["params.patchsize"].astype(int)
    auc_plot = runs_df.hvplot.line(
        x="patchsize",
        y="metrics.anomaly_auc",
        grid=True,
    )
    return auc_plot


def main_ablation_patchsize():
    dloads = get_dataloaders(ablation_patchsize_exp_name)

    runs_df = mlflow.search_runs(experiment_names=[ablation_patchsize_exp_name])
    runs_df["patchsize"] = runs_df["params.patchsize"].astype(int)

    patchsize_w = pn.widgets.Select(
        name="patchsize", value=9, options=runs_df["patchsize"].tolist()
    )

    @pn.depends(patchsize_w)
    def view(patchsize: int):
        """
        >>> patchsize = 9
        """
        output = pn.Column(patchsize_w)
        run_id = runs_df[runs_df.patchsize == patchsize].run_id.item()
        df = util.load_checkpoint_df(run_id, dloads.test, ("100",))
        metric_keys = sorted(df.metrics.iloc[0])
        output.append(
            img_table(
                df,
                metric_keys,
                get_expand_content_func(ablation_patchsize_exp_name, run_id),
            )
        )

        return output

    return pn.Column(patchcore_patchsize_auc(), view)


def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def main_ablation_patchsize_gradcams():
    """Creates 3 plots for demonstrating the effects of patchsize and anomaly maps postprocessing"""
    dloads = get_dataloaders(ablation_patchsize_exp_name)

    runs_df = mlflow.search_runs(experiment_names=[ablation_patchsize_exp_name])
    eval_df = pd.concat(
        [
            util.load_checkpoint_df(run_id, dloads.test, ("100",)).assign(run_id=run_id)
            for run_id in runs_df.run_id
        ]
    )

    patchsizes = [1, 3, 5, 9, 15, 25]
    id_to_patchsize = runs_df.set_index("run_id")["params.patchsize"].astype(int)
    patchsize_to_plot = {}
    for run_id, df in eval_df.groupby("run_id"):
        patchsize = id_to_patchsize[run_id]
        if patchsize not in patchsizes:
            continue
        largest_inliers = df.sample(4, random_state=42)
        func = get_expand_content_func(ablation_patchsize_exp_name, run_id)
        anom_imgs = [func(row) for row in largest_inliers.itertuples()]
        patchsize_to_plot[patchsize] = anom_imgs

    patchsize_example_plots = pn.Column(
        *[
            pn.Row(
                *[plot[-1] for plot in patchsize_to_plot[patchsize]],
                f'<span style="font-size:150px;">{patchsize}</span>',
            )
            for patchsize in patchsizes
        ]
    )

    def postprocessing():
        from patchcore import patchcore

        dloads = get_dataloaders(ablation_patchsize_exp_name)
        model: patchcore.PatchCore = util.load_model_cached(run_id, train_pct=100).to(
            "cuda"
        )
        transform = models.transform_dict["resnet18_crop_only"]

        img_raw = anom_imgs[0][0].object
        processed_img = dloads.test.dataset.transform(img_raw).to("cuda").unsqueeze(0)
        img: Image = transform(img_raw)

        model.predict(processed_img)
        s1 = model._patch_scores
        s2 = (
            F.interpolate(
                torch.from_numpy(s1).unsqueeze(1),
                size=img.size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .numpy()
        )
        s3 = ndimage.gaussian_filter(s2, sigma=4)
        anomaly_stages = hv.Layout(
            [
                pd.DataFrame(s.squeeze()).hvplot.heatmap(
                    colorbar=False,
                    yaxis=False,
                    aspect=1,
                    shared_axes=False,
                    xaxis=False,
                )
                for s in [s1, s2, s3]
            ]
        )
        return anomaly_stages

    def grid_img():
        step = 512 / 14
        arr = np.array(anom_imgs[0][1].object)
        for i in range(1, 14):
            idx = int(step * i)
            arr[:, idx, :] = 0
            arr[idx, :, :] = 0
        grid_img = Image.fromarray(arr)
        return grid_img

    return pn.Column(patchsize_example_plots, postprocessing(), grid_img())


@functools.cache
def mlflow_load_dict_cached(uri: str):
    return mlflow.artifacts.load_dict(uri)


def confusion_matrices():
    """
    Plots the average confusion matrix of every 1% of TPR for the ROC curve.
    Uses the patchcore model with imagenet weights
    """
    runs_df = mlflow_summary.load_runs_df()
    runs_df = mlflow_summary.exp_name_to_run_info(runs_df)
    runs_df = runs_df.map(util.try_float)

    metric = "metrics.score"

    runs_df["AUC"] = runs_df["metrics.anomaly_auc"] * 100
    runs_df["Anomaly dataset"] = (
        runs_df["clf_ds"].str.replace("condition-", "").str.replace("images-", "")
    )
    runs_df["Classifier"] = runs_df["model"].str.replace("resnet18_", "")

    imagenet_exps = [
        e
        for e in mlflow.search_experiments(
            filter_string='name LIKE "test_%imagenet_patchcore"'
        )
    ]
    imagenet_exps.sort(key=lambda x: x.name, reverse=True)
    confusion_plots = []
    roc_plots = []
    for exp in imagenet_exps:
        exp_name = exp.name
        ds_name = (
            exp_name.split("__")[1].replace("condition-", "").replace("images-", "")
        )
        try:
            run_id = util.run_id_from_exp_name(exp_name)
        except ValueError:
            print("skipping", exp_name)
            continue

        # Get the artifact foo.json from the run
        uri = mlflow.get_run(run_id).info.artifact_uri
        evals = mlflow_load_dict_cached(f"{uri}/evals_100.json")

        df = pd.json_normalize(evals)
        labels_w_counts = {
            label: f"{label} [{count}]"
            for label, count in df["class_label"].value_counts().items()
        }

        df["y"] = df["class_label"].ne("good") & df["class_label"].ne("inlier")
        assert df["y"].any()
        roc_df = pd.DataFrame(
            metrics.roc_curve(df["y"], df[metric]), index=["fpr", "tpr", "thresh"]
        ).T
        best_roc_df = roc_df.drop_duplicates("fpr", keep="last", ignore_index=True)
        roc_plots.append(
            hv.Overlay(
                [hv.VLine(pct).opts(alpha=0.25) for pct in np.linspace(0, 1, 101)]
            )
            * roc_df.hvplot.line(x="fpr", y="tpr", title=exp_name).opts(
                toolbar=None,
                title=ds_name,
                yformatter=pct_fmt(),
                xformatter=pct_fmt(),
                fontsize={"labels": 14, "xticks": 11, "yticks": 11},
            )
            * best_roc_df.hvplot.points(s=50).opts(ylabel="TPR", xlabel="FPR")
        )
        roc_sampled_df = pd.concat(
            [
                df.assign(
                    pred=(
                        df[metric] >= roc_df.loc[roc_df.fpr.searchsorted(pct)].thresh
                    ).map({True: "anomaly", False: "good"})
                )
                for pct in np.linspace(0, 1, 101)
            ]
        )

        confusion_matrix = pd.crosstab(
            roc_sampled_df["class_label"], roc_sampled_df["pred"], normalize="index"
        )
        confusion_matrix.index = (
            confusion_matrix.index.map(labels_w_counts)
            .str.replace("-or-", "/")
            .str.replace("-off", "")
        )
        confusion_plot = confusion_matrix.hvplot.heatmap(
            clabel="Predicted fraction",
            clim=(0, 1),
            colorbar=False,
            title=ds_name,
            shared_axes=False,
            fontsize={"labels": 14, "xticks": 14, "yticks": 12},
        ).opts(toolbar=None)
        labels = hv.Labels(confusion_plot).opts(
            text_color="orange", text_font_size="14pt"
        )
        labels.vdims[0].value_format = lambda x: f"{x:.0%}"
        confusion_plots.append((confusion_plot * labels).opts(toolbar=None))

    for plot in confusion_plots:
        plot.opts(toolbar=None, shared_axes=False)
        plot.opts(hv.opts.HeatMap(width=300, cformatter=pct_fmt()))

    cm_scavport_plot, *cm_condition_plots = confusion_plots
    cm_scavport_plot.opts(hv.opts.HeatMap(colorbar=True))
    cm_condition_plots = (
        hv.Layout(cm_condition_plots).opts(shared_axes=False, toolbar=None).cols(2)
    )
    cm_condition_plots

    roc_scavport_plot, *roc_condition_plots = roc_plots
    roc_scavport_plot.opts(toolbar=None, width=300)
    roc_condition_plots = (
        hv.Layout(roc_condition_plots)
        .opts(shared_axes=False, toolbar=None)
        .opts(hv.opts.Curve(width=300))
        .cols(2)
    )
    roc_condition_plots

    # hv.save(cm_scavport_plot, DOCS_PATH / "confusion_matrix_scavport.png")
    # hv.save(cm_condition_plots, DOCS_PATH / "confusion_matrix_condition.png")

    # hv.save(roc_scavport_plot, DOCS_PATH / "roc_avg_scavport.png")
    # hv.save(roc_condition_plots, DOCS_PATH / "roc_avg_condition.png")

    return pn.Column(
        pn.Row(cm_scavport_plot, cm_condition_plots),
        pn.Row(roc_scavport_plot, roc_condition_plots),
    )


def example_imgs():
    # DS_1_NAME="images-scavengeport-overview"
    # DS_2_NAME="lock-condition"
    # DS_3_NAME="ring-condition"
    # DS_4_NAME="ring-surface-condition"
    datasets = [
        "embed__scavport",
        "embed__vesselarchive",
        "anomaly__scavport",
        "anomaly__condition-ring-condition",
        "anomaly__condition-ring-surface-condition",
        "anomaly__condition-lock-condition",
        "anomaly__condition-images-scavengeport-overview",
    ]
    out = pn.GridBox(ncols=5)
    for ds in datasets:
        dload = get_dataloaders(ds).test
        ds = ds.replace("images-", "").replace("condition-", "")
        idx2cls = util.idx_to_class(dload)
        seen = set()
        for path, cls_idx in dload.dataset.imgs[::-1]:
            if cls_idx in seen:
                continue
            seen.add(cls_idx)
            img = Image.open(path).resize((256, 256))
            out.append(pn.Column(f"#### {ds} \n [{idx2cls[cls_idx]}]", img))

    # pn.serve(pn.GridBox(*out[:13], ncols=5))
    # pn.serve(pn.GridBox(*out[13:], ncols=4))
    return out


def feature_comparison():
    experiment_widget = pn.widgets.Select(
        value=None,
        options=[None, *name_to_experiment],
        name="experiment",
    )
    runs_df = mlflow_summary.load_runs_df()
    runs_df = mlflow_summary.exp_name_to_run_info(runs_df)
    runs_df = runs_df.map(util.try_float)

    @pn.depends(experiment_widget)
    def _inner(exp_name):
        df = runs_df[runs_df.exp_name == exp_name] if exp_name else runs_df
        return df.hvplot.explorer()

    return pn.Column(experiment_widget, _inner)


def effective_patchsize():
    runs_df = mlflow_summary.load_runs_df()
    runs_df = mlflow_summary.exp_name_to_run_info(runs_df)
    runs_df = runs_df.map(util.try_float)
    runs_df["effective_patchsize"] = runs_df["params.patchsize"] * runs_df[
        "params.layers_to_extract_from"
    ].map(
        lambda x: None
        if not x
        else 14
        if "layer3" in x
        else 28
        if "layer2" in x
        else 56
        if "layer1" in x
        else None
    )
    runs_df["effective_patchsize"] = np.where(
        runs_df["effective_patchsize"] > 448, 448, runs_df["effective_patchsize"]
    )

    auc = "metrics.anomaly_auc"
    hexbin = runs_df.assign(**{auc: lambda df: df[auc] * 100}).hvplot(
        yformatter="%.0f%%",
        title="Optimal effective patchsize search",
        ylabel="AUC",
        xlabel="Patch size",
        cmap="HighContrast_r",
        cnorm="log",
        colorbar=True,
        height=500,
        xlim=(0, 460),
        kind="hexbin",
        width=500,
        x="effective_patchsize",
        y=auc,
        legend="bottom_right",
        hover_cols=["run_id"],
        fontsize=fontsize,
    )
    patch_size_search_plot = hexbin * hv.Points(hexbin).opts(size=1, color="black")
    patch_size_search_plot
    # hv.save(
    #     patch_size_search_plot.opts(toolbar=None),
    #     DOCS_PATH / "patch_size_search_plot.png",
    # )
    return patch_size_search_plot


def layers_to_extract_from():
    runs_df = mlflow_summary.load_runs_df()
    runs_df = mlflow_summary.exp_name_to_run_info(runs_df)
    runs_df = runs_df.map(util.try_float)

    runs_df["AUC"] = runs_df["metrics.anomaly_auc"] * 100
    runs_df["Anomaly dataset"] = (
        runs_df["clf_ds"].str.replace("condition-", "").str.replace("images-", "")
    )
    runs_df["Classifier"] = runs_df["model"].str.replace("resnet18_", "")

    plots = (
        runs_df.sort_values("params.layers_to_extract_from")
        .dropna(subset="params.layers_to_extract_from")
        .hvplot(
            by="params.layers_to_extract_from",
            cmap="HighContrast_r",
            cnorm="log",
            colorbar=True,
            row="Anomaly dataset",
            col="Classifier",
            kind="box",
            y="AUC",
            yformatter="%.0f%%",
            legend=None,
            toolbar=None,
            tools=[],
            show_legend=False,
            width=1000,
            height=1000,
            ylabel="AUC",
            xlabel="Layers",
            rot=90,
            hooks=[remove_bokeh_logo],
        )
        .opts(toolbar=None, width=1000, height=1000)
    )

    # new_plots = {}
    for key, plot in plots.items():
        plot.opts(toolbar=None)
    #     new_plots[key] = plot.options(outlier_alpha=0) * hv.Scatter(plot).opts(
    #         color="black", size=1
    #     ).opts(jitter=0.2)

    # new_plots = hv.GridSpace(new_plots, ["Anomaly dataset", "Classifier"])

    # We have to render it this way to avoid the logo
    plot = hv.render(plots)
    # Disable the toolbar
    plot.toolbar.tools.clear()
    plot.toolbar.logo = None
    # bokeh.io.export_png(
    #     plot,
    #     filename=DOCS_PATH / "layers_to_extract_from.png",
    # )
    return plots


def remove_bokeh_logo(plot, element):
    plot.state.toolbar.logo = None


def embedding_loss_by_anomalous():
    def batch_losses(model, imgs) -> list[float]:
        if type(model) == models.SupConResNet:
            return util.sup_con_loss(model, imgs, None).tolist()
        elif type(model) == models.EncoderBottleneckDecoderResNet:
            input_layers, output_layers = model(imgs)
            return util.batch_layer_loss(input_layers, output_layers).tolist()
        else:
            raise NotImplementedError(type(model))

    scavport_anomaly = get_dataloaders("anomaly_scavport").test

    data = []
    exp_name_fmt = "embed_loss__{bb_dataset}__resnet18_{bb_model}__imagenet"
    for bb_model_name in ["self-sup-con", "T-S"]:
        if bb_model_name == "self-sup-con":
            scavport_anomaly.dataset.transform = util.add_double_aug(
                util.add_randaug(models.transform_dict["resnet18"])
            )
        else:
            scavport_anomaly.dataset.transform = models.transform_dict["resnet18"]

        for bb_dataset in ["scavport", "vesselarchive"]:
            exp_name = exp_name_fmt.format(
                bb_dataset=bb_dataset, bb_model=bb_model_name
            )
            model = util.load_model_cached(exp_name, 100).to(config.DEVICE)
            for img, classes in scavport_anomaly:
                inp, out = model(img.to(config.DEVICE))
                inp[-1][4, :, :, :].mean(-1).mean(-1).median().detach().cpu().numpy()
                pd.DataFrame(
                    inp[2][7, 10, :, :].detach().cpu().numpy()
                ).hvplot.heatmap()
                pd.DataFrame(
                    inp[2][7, 10, :, :].detach().cpu().numpy()
                ).hvplot.heatmap()
                losses = batch_losses(model, img.to(config.DEVICE))
                data.extend(
                    zip(
                        losses,
                        classes.tolist(),
                        itertools.repeat(bb_model_name),
                        itertools.repeat(bb_dataset),
                    )
                )
    embed_loss_by_anomaly_df = (
        pd.DataFrame(data, columns=["loss", "anomaly", "bb_model", "bb_dataset"])
        .groupby(
            [
                "bb_model",
                "anomaly",
                "bb_dataset",
            ]
        )
        .agg("mean")
    )
    print(embed_loss_by_anomaly_df.to_latex(float_format="%.1e").replace("_", "-"))

    return embed_loss_by_anomaly_df


def supcon_to_rd4ad_encoder(model: models.SupConResNet):
    rd4ad_model = rd4ad.resnet_models.resnet18()
    failures = rd4ad_model.teacher_encoder.load_state_dict(
        model.encoder.state_dict(), strict=False
    )
    assert (
        failures.missing_keys == ["fc.weight", "fc.bias"]
        and not failures.unexpected_keys
    )
    return rd4ad_model.teacher_encoder


def mean_or_max_pixel_agg():
    runs_df = mlflow_summary.load_runs_df()
    runs_df = mlflow_summary.exp_name_to_run_info(runs_df)
    runs_df = runs_df.map(util.try_float)
    exp_name = "optuna_anomaly__scavport__resnet18_T-S__imagenet"
    run_df = runs_df[runs_df.exp_name == f"{exp_name}"]
    boxplot = run_df.hvplot.box(
        "metrics.anomaly_auc",
        by="params.pixel_agg_func",
        width=210,
        height=210,
        yformatter=pct_fmt(),
        ylabel="AUC",
        xlabel="Anomaly map agg. function",
        title="ImageNet T-S\nScavPort Anomaly",
        ylim=(0.6, 0.8),
    ).opts(toolbar=None)
    hv.save(boxplot, DOCS_PATH / "pixel_agg_func.png")
    return boxplot


def backbone_weight_amplitudes():
    experiments = mlflow.search_experiments()
    backbones = {
        e.name: util.load_model_cached(e.name, 100)
        for e in experiments
        if e.name.startswith("embed_loss")  # and "T-S" in e.name
    }
    backbones = {
        k: (getattr(v, "teacher_encoder", None) or supcon_to_rd4ad_encoder(v)).to(
            config.DEVICE
        )
        for k, v in backbones.items()
    }
    baseline = rd4ad.resnet_models.resnet18().teacher_encoder.to(config.DEVICE)
    baseline_params = dict(baseline.named_parameters())

    data = []
    for bb_name, bb in backbones.items():
        for param_name, param in bb.named_parameters():
            baseline_param = baseline_params[param_name]
            param_diff = (param - baseline_param).abs() / baseline_param
            data.append([bb_name, param_name, param_diff.mean().item()])
    df = pd.DataFrame(data, columns=["bb_name", "param_name", "mean_diff"])
    logbins = np.logspace(-3, 1, 100)
    cdf = df.hvplot.hist(
        "mean_diff",
        by="bb_name",
        bins=logbins,
        logx=True,
        cumulative=True,
    ).dframe()
    cdf["frac"] = cdf.groupby("bb_name")["mean_diff_count"].transform(
        lambda x: x / x.max()
    )
    return cdf.hvplot.line(
        x="mean_diff",
        y="frac",
        by="bb_name",
        legend="bottom_right",
        title="CDF of Backbone weight amplitudes",
        ylabel="Cumulative fraction of weights",
        xlabel="Weight amplitude",
        yformatter=pct_fmt(),
    ).opts(toolbar=None)


# def backbone_activation_amplitude():
#     dfs = []
#     for model_name, model in backbones.items():
#         for img, class_ in scavport_loader:
#             third_layer_activations = (
#                 model(img.to(config.DEVICE))[-1].detach().cpu().flatten()
#             )
#             data = pd.DataFrame(third_layer_activations.abs(), columns=["activation"])
#             dfs.append(data.assign(model_name=model_name))
#     df = pd.concat(dfs)
#     sample = df.sample(1_000_000)
#     sample["activation"] = sample["activation"].clip(0, 0.8)

#     plots = sample.hvplot.hist("activation", cumulative=True, by="model_name", bins=100)
#     dframes = []
#     for k, v in plots.items():
#         dframe = v.dframe().assign(
#             frac=lambda df: df["activation_count"] * 100 / df["activation_count"].max()
#         )
#         dframes.append(dframe.assign(model_name=k))
#     activations_cdf = (
#         pd.concat(dframes)
#         .hvplot.line(
#             x="activation",
#             y="frac",
#             by="model_name",
#             legend="bottom_right",
#             title="CDF of Backbone activations after the 3rd ResNet18 layer",
#             ylabel="Cumulative fraction of weights",
#             xlabel="Neuron activation amplitude (Capped at 0.8)",
#             yformatter="%.0f%%",
#             ylim=(0, 100),
#         )
#         .opts(toolbar=None)
#     )
#     hv.save(
#         activations_cdf,
#         DOCS_PATH / "backbone_activations_cdf.png",
#     )
#     return activations_cdf


def imagenet_classes():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    scavport_loader = get_dataloaders("anomaly_scavport").test
    categories = dict(enumerate(ResNet18_Weights.DEFAULT.meta["categories"]))

    scavport_results = []
    for img, class_ in scavport_loader:
        preds = model(img).softmax(0).max(0)
        scavport_results.extend(
            zip(preds.indices.tolist(), class_.tolist(), preds.values.tolist())
        )
    df = pd.DataFrame(scavport_results, columns=["pred", "class", "confidence"])
    df["pred"] = df["pred"].map(categories)
    scavport_classes = (
        df.groupby("class")
        .apply(lambda g: g["pred"].value_counts(normalize=True))
        .reset_index()
        .to_latex(index=False, float_format="%.2f")
    )

    archive_loader = get_dataloaders("vessel").test
    archive_results = []
    for img, class_ in archive_loader:
        preds = model(img).softmax(0).max(0)
        archive_results.extend(
            zip(preds.indices.tolist(), class_.tolist(), preds.values.tolist())
        )
    df = pd.DataFrame(archive_results, columns=["pred", "class", "confidence"])
    df["pred"] = df["pred"].map(categories)
    archive_classes = (
        df["pred"]
        .value_counts(normalize=True)
        .reset_index()
        .to_latex(index=False, float_format="%.2f")
    )

    pn.Column(
        "Avg Confidence",
        f"{np.mean(df['confidence']):.0%}",
        pn.Row(
            pn.Column("Scavport", scavport_classes),
            pn.Column("Archive", archive_classes),
        ),
    )


pn.serve(
    {
        "1_anomaly": main_anomaly,
        "1_embed": main_embed,
        "1_summary": main_summary,
        "1_confusion_matrices": confusion_matrices,
        "1_example_imgs": example_imgs,
        "2_layers_to_extract_from": layers_to_extract_from,
        "2_effective_patchsize": effective_patchsize,
        "2_feature_comparison": feature_comparison,
        "2_ablation_patchsize": main_ablation_patchsize,
        "3_imagenet_classes": imagenet_classes,
        "3_patchsize_gradcams_example": main_ablation_patchsize_gradcams,
        "3_embedding_loss_by_anomalous": embedding_loss_by_anomalous,
        "3_backbone_weight_amplitudes": backbone_weight_amplitudes,
    }
)
