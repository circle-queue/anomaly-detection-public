import os
import platform
from importlib.resources import files
from pathlib import Path

import numpy as np

# import torch

IS_CLOUD = os.getlogin() == "ec2-user"

# assert torch.cuda.is_available(), "CUDA is not available. Check your setup."
LOCAL = platform.system() == "Windows"
REPO_DATA_ROOT: Path = files("anomaly_detection") / "data"
DATA_ROOT = Path("E:\\" if LOCAL else "/home/ec2-user/SageMaker/")
SCAV_PORT_ROOT = DATA_ROOT / "scav-port-data"
VESSEL_ARCHIVE_ROOT = DATA_ROOT / "vessel-archive"
CONDITION_ROOT = DATA_ROOT / "condition-ds"
REPO_ROOT = Path(
    r"E:\anomaly-detection" if LOCAL else "/home/ec2-user/SageMaker/anomaly-detection"
)
# DATA_ROOT, REPO_ROOT = (
#     Path(r"C:\Users\AES9RSQ\scav-port-data"),
#     Path(r"C:\Users\AES9RSQ\Documents\repo\anomaly-detection"),
# )
DEVICE = "cuda"
CKP_PATH_TEMPLATE = str(REPO_ROOT / "checkpoints/{exp_name}/{train_pct}.pth")
EVAL_PATH_TEMPLATE = str(Path(CKP_PATH_TEMPLATE).with_suffix(".json"))
OPTUNA_PATH_TEMPLATE = f"sqlite:///{REPO_ROOT}/optuna/{{}}.sqlite3"
Path(REPO_ROOT / "optuna").mkdir(exist_ok=True)

TRAIN_CLASSES = sorted(p.name for p in (SCAV_PORT_ROOT / "train").iterdir())
EVAL_CLASSES = sorted(p.name for p in (SCAV_PORT_ROOT / "dev").iterdir())

SCAVPORT_CLF_CLASSES = [
    "liner",
    "piston-ring-overview",
    "topland",
    "piston-top",
    "single-piston-ring",
    "skirt",
    "scavange-box",
    "piston-rod",
]

CKP_PCTS = list(np.logspace(0, 2, 10).round().astype(int))
DIMO_ROOT = Path(r"D:\dimo_small")

TRACK_LOSS = None

MLFLOW_URI = "http://127.0.0.1:5000"

EVAL_GROUPING = ["task", "clf_ds", "model", "bb_model"]
EVAL_DERIVED = ["bb_method", "bb_ds", "methods", "bb_ds_methods", "bb_ds+clf_method"]
