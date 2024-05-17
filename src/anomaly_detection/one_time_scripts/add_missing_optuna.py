from pathlib import Path

import mlflow
from anomaly_detection import config

exp_id_to_name = {e.experiment_id: e.name for e in mlflow.search_experiments()}

mlflow.set_tracking_uri(config.REPO_DATA_ROOT.parents[2] / "mlruns")
all_runs = mlflow.search_runs(search_all_experiments=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

all_runs = all_runs.assign(exp_name=lambda df: df.experiment_id.map(exp_id_to_name))

for file in Path("optuna").iterdir():
    if file.suffix != ".sqlite3":
        continue
    print(file)
    exp_name = file.stem
    # Find all parent runs and upload the file as an artifact
    parent_runs = all_runs.query(
        f'exp_name == "{exp_name}" and `tags.mlflow.parentRunId`.isnull()'
    )
    for run in parent_runs.run_id:
        with mlflow.start_run(run_id=run):
            mlflow.log_artifact(str(file))
