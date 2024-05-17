import json
import random
from pathlib import Path
from typing import NamedTuple

import anomaly_detection
import boto3
import pandas as pd
from anomaly_detection import config
from anomaly_detection.one_time_scripts.copy_to_s3 import BUCKET, s3_list_files
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def s3_download(s3_to_local_path: tuple[str, Path]):
    s3_file, local_path = s3_to_local_path
    if local_path.exists():
        return
    local_path.parent.mkdir(exist_ok=True, parents=True)

    s3 = boto3.client("s3")
    s3.download_file(BUCKET,s3_file,local_path)


if __name__ == '__main__':
    s3_files = s3_list_files()

    s3_to_local_path = [(s3_file, config.DATA_ROOT / s3_file) for s3_file in s3_files]
    s3_to_local_path.sort(key=lambda _: random.random())
    process_map(s3_download,s3_to_local_path, chunksize=100)

