import shutil

import mlflow
import mlflow.tracking.fluent
from anomaly_detection import config

exp_id_to_name = {e.experiment_id: e.name for e in mlflow.search_experiments()}

root = config.REPO_DATA_ROOT.parents[2] / "mlruns"
mlflow.set_tracking_uri(root)
for experiment in mlflow.search_experiments():
    if all(key not in experiment.name for key in ["optuna", "ablation", "Default"]):
        runs = mlflow.search_runs([experiment.experiment_id])
        assert len(runs) < 2, f"Multiple runs found for {experiment.name}"
