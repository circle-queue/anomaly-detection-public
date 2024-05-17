import mlflow
import mlflow.entities
import mlflow.tracking.fluent
from anomaly_detection import config

exp_id_to_name = {e.experiment_id: e.name for e in mlflow.search_experiments()}

root = config.REPO_DATA_ROOT.parents[2] / "mlruns"
mlflow.set_tracking_uri(root)

for experiment in mlflow.search_experiments():
    if "T-S" in experiment.name:
        print(experiment.name)
        # exp_root = root / experiment.experiment_id
        # assert exp_root.exists()
        # print("Deleting", experiment.name)
        # exp_root.rename(config.DATA_ROOT / "mlruns_bak" / experiment.experiment_id)

        for run_id in mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
        ).run_id:
            print(run_id)
            # path = root / experiment.experiment_id / run_id
            # dst = config.DATA_ROOT / "mlruns_bak" / experiment.experiment_id / run_id
            # dst.parent.mkdir(parents=True, exist_ok=True)
            # path.rename(dst)
            # # mlflow.tracking.MlflowClient().restore_run(run_id)
            mlflow.delete_run(run_id)
