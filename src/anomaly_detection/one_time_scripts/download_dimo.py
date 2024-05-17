# https://pderoovere.github.io/dimo/downloads/
import io
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
from anomaly_detection.config import DATA_ROOT
from tqdm.auto import tqdm

DIMO_ROOT = DATA_ROOT / "dimo"

BASE_URL = "https://datasets.ilabt.imec.be/dimo/tar/dimo.tar"
DIMO_SMALL = "https://datasets.ilabt.imec.be/dimo/tar/dimo_small.tar"  # 64 GB


def download_large_file(url) -> Path:
    tmp_dst = Path(tempfile.gettempdir()) / Path(url).name
    # if tmp_dst.exists():
    #     print(f"{tmp_dst} already exists, skipping download")
    #     return tmp_dst

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {url} to {tmp_dst} ...",
            total=int(r.headers["Content-Length"]),
        )
        with open(tmp_dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=10 * 8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return tmp_dst


if __name__ == "__main__":
    local_path = download_large_file(url=DIMO_SMALL)

    print("Extracting...")
    with tarfile.open(local_path) as f:
        f.extractall(DIMO_ROOT)
