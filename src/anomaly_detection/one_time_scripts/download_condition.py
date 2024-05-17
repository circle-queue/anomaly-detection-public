from pathlib import Path
from typing import NamedTuple

import numpy as np
from anomaly_detection.config import DATA_ROOT
from PIL import Image
from tqdm.auto import tqdm


def cp(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = Image.open(src)
    img.thumbnail((512, 512))  # No larger than 512x512
    img.save(dst)


def create_random_mask(dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # if dst.exists():
    #     return
    arr = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    Image.fromarray(arr).save(dst)


class Dataset(NamedTuple):
    root: Path
    good_label: str
    bad_labels: list[str]

    def download(self):
        ds_name = self.root.name.lower().replace(" ", "-")
        condition_root = DATA_ROOT / "condition-ds" / ds_name

        good_paths = list(self.root.glob(f"{self.good_label}/*.jpg"))
        bad_labels_paths = {
            label: list(self.root.glob(f"{label}/*.jpg")) for label in self.bad_labels
        }
        assert good_paths and bad_labels_paths

        idx = 0
        for drive_path in tqdm(good_paths, desc=f"Copying {self.root} good images"):
            idx = (idx + 1) % 3
            split = {0: "train", 1: "dev", 2: "test"}[idx]
            # First copy is from network drive, then we can copy from local
            local_path = condition_root / f"{split}/good" / drive_path.name
            cp(drive_path, local_path)

        for group, bad_paths in tqdm(
            list(bad_labels_paths.items()), desc=f"Copying {self.root} bad images"
        ):
            group = group.lower().replace(" ", "-")
            for drive_path in bad_paths:
                idx = (idx + 1) % 2
                split = {0: "dev", 1: "test"}[idx]
                local_path = condition_root / f"{split}/{group}" / drive_path.name
                cp(drive_path, local_path)
                create_random_mask(
                    condition_root
                    / f"ground_truth/{group}"
                    / drive_path.with_suffix(".png").name
                )


# Drive root:
BANE_SRC = Path("G:\wear-database\ScavengePortInspection\BANE new pictures")

lock_condition = Dataset(
    BANE_SRC / "Lock Condition",
    "Intact",
    ["Visible Cracks", "Coating Missing or peeled off", "Burn Mark", "Broken"],
)
ring_condition = Dataset(
    BANE_SRC / "Ring Condition",
    "Intact",
    ["Collapsed", "Broken", "Missing"],
)
ring_surface_condition = Dataset(
    BANE_SRC / "Ring Surface Condition",
    "Clean and smooth",
    [
        "Coating Cracks",
        "Coating Peel-off",
        "Embedded iron",
        "scuffing",
        "signs of abrasive wear",
        "signs of adhesive wear",
    ],
)
overview_condition = Dataset(
    Path("G:\wear-database\ScavengePortInspection\Images scavengeport overview"),
    "Normal",
    ["abnormal"],  # , "oil not wiped", "oil wiped out"],
)
for ds in [lock_condition, ring_condition, ring_surface_condition, overview_condition]:
    ds.download()
