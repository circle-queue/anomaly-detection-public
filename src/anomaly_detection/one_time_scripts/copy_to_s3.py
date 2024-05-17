import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import anomaly_detection
import boto3
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

REPO_DATA_ROOT: Path = files(anomaly_detection) / "data"
ORIG_LABEL_COLS = [
    "Piston Ring Overview",  # 2300
    "Liner",  # 1800
    "Single Piston Ring",  # 1500
    "topland",  # 650
    "piston top",  # 330
    "skirt",  # 330
    "scavange box",  # 250
    "piston rod",  # 45 images
    # "scavange port",  # UNK imgs
]
DATA_ROOT = Path("E:\\")
SCAV_PORT_ROOT = DATA_ROOT / "scav-port-data"
VESSEL_ARCHIVE_ROOT = DATA_ROOT / "vessel-archive"

IGNORE_SUFFIXES = [".clip_features", ".features", ".pt"]
IMGS_ROOT = Path(r"E:\component-classifier\src\component_classifier\data\images")

BUCKET = "sommerlund-thesis"


def add_label_and_target(df: pd.DataFrame) -> pd.DataFrame:
    # Add labels by converting a string to a dict, and selecting rows with exactly one label
    df["type_dict"] = df.type.apply(lambda x: eval(x) if pd.notna(x) else {})

    labels_df = pd.json_normalize(df["type_dict"]).drop(
        columns=["cylinder nr visable", "cylinder nr", "rotate"]
    )
    df[labels_df.columns] = labels_df.fillna(0).astype(float).values

    df = df[(df[ORIG_LABEL_COLS].sum(axis=1) == 1)].copy()
    df["label"] = df[ORIG_LABEL_COLS].idxmax(axis=1).str.replace(" ", "-").str.lower()

    label_to_target = {label: i for i, label in enumerate(sorted(df.label.unique()))}
    df["target"] = df.label.map(label_to_target).astype(float)
    return df


def build_inlier_outlier_folder():
    img_id_to_path = {
        path.stem: path
        for path in IMGS_ROOT.iterdir()
        if path.suffix not in IGNORE_SUFFIXES
    }

    df = (
        pd.read_csv(files(anomaly_detection) / "data/annotations.csv")
        .rename(columns={"annot": "target"})
        .replace({"label": "inlier", "no-label": "outlier"})
    )

    size = int(len(df) * 0.50)
    df_dev, df_test = train_test_split(
        df, test_size=size, stratify=df["target"], random_state=42
    )

    dataframes = [("dev", df_dev), ("test", df_test)]
    save_named_dfs(dataframes, img_id_to_path)


def build_8_class_folder():
    ids_exist = {
        int(f.stem) for f in IMGS_ROOT.glob("*") if f.suffix not in IGNORE_SUFFIXES
    }

    data_root = files(anomaly_detection) / "data/images_db_mariel.tsv"
    df = pd.read_csv(data_root, sep="\t", encoding="latin1")
    df = df[df.id.isin(ids_exist)]

    df = add_label_and_target(df)
    df["target"] = df["label"]

    size = int(len(df) * 0.15)
    df_train_test, df_dev = train_test_split(
        df, test_size=size, stratify=df["target"], random_state=42
    )

    df_train, df_test = train_test_split(
        df_train_test, test_size=size, stratify=df_train_test["target"], random_state=42
    )

    img_id_to_path = {
        path.stem: path
        for path in IMGS_ROOT.iterdir()
        if path.suffix not in IGNORE_SUFFIXES
    }

    dataframes = [("train", df_train), ("dev", df_dev), ("test", df_test)]
    save_named_dfs(dataframes, img_id_to_path)


def save_named_dfs(
    dataframes: list[tuple[str, pd.DataFrame]], img_id_to_path: dict[str, Path]
):
    for split, df in dataframes:
        for row in df.itertuples():
            # Rescale images to 224x224
            src = img_id_to_path[str(row.id)]
            dst = SCAV_PORT_ROOT / f"{split}/{row.target}/{row.id}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(src).convert("RGB")
            img.thumbnail((512, 512))  # No larger than 512x512
            img.save(dst)


def s3_list_files() -> list[str]:
    s3 = boto3.client("s3")
    paths = []

    response = s3.list_objects_v2(Bucket=BUCKET)
    while True:
        paths += [obj["Key"] for obj in response["Contents"]]
        if not response.get("IsTruncated"):
            break

        response = s3.list_objects_v2(
            Bucket=BUCKET, ContinuationToken=response["NextContinuationToken"]
        )
    return paths


def s3_upload(file: Path, seen: set = set()):
    s3 = boto3.client("s3")
    if not seen:
        seen.update(s3_list_files())

    dst = file.relative_to(DATA_ROOT).as_posix()
    if dst in seen:
        return

    s3.upload_file(str(file), BUCKET, dst)


class Download(NamedTuple):
    id: int
    b_drive_src: Path
    local_dst: Path
    s3_dst: str


def download(id_file: tuple[int, Path]) -> Download | None:
    id_, file = id_file
    label = "dev" if 0 == hash(id_) % 10 else "train"
    dst = VESSEL_ARCHIVE_ROOT / f"{label}/{id_}.jpg"
    data = Download(id_, file, dst, dst.relative_to(DATA_ROOT).as_posix())
    if dst.exists():
        return data

    try:
        Image.open(file).verify()
        img = Image.open(file).convert("RGB")  # RGBA doesn't work with jpg
    except Exception:
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.thumbnail((512, 512))
    img.save(dst)
    return data


def build_operations_folder():
    valid_img_suffix = set(Image.ID) | {"JPG"}

    imo_to_path: dict = json.loads(
        Path(r"B:/Vessel_Archive/vessel_archive_directory.json").read_text()
    )

    archive_files_cache = DATA_ROOT / "archive_files.json"
    if archive_files_cache.exists():
        archive_files = list(map(Path, json.loads(archive_files_cache.read_text())))
    else:
        archive_files = [
            file
            for engine_dir in tqdm(imo_to_path.values(), smoothing=0)
            for file in Path(engine_dir).rglob("*")
        ]
        archive_files_cache.write_text(
            json.dumps([str(file) for file in archive_files])
        )

    id_to_img_path = {
        abs(hash(file)): file
        for file in archive_files
        if file.suffix.upper().strip(".") in valid_img_suffix
    }
    print(f"{len(id_to_img_path) = }")  # 128762

    # suffixes = collections.Counter(file.suffix.lower() for file in archive_files)
    # print(suffixes.most_common(25))
    # [('.jpg', 127741), ('.pdf', 25147), ('.csv', 24521), ('', 5478), ('.cgm', 4718), ('.emf', 4716), ('.xlsx', 2077), ('.dat', 1972), ('.docx', 1246), ('.xls', 747), ('.manbw-spaf', 640), ('.png', 636), ('.xlsm', 624), ('.db', 573), ('.eds', 566), ('.zip', 412), ('.msg', 348), ('.rci', 335), ('.tsv', 261), ('.xml', 220), ('.jpeg', 210), ('.bmp', 165), ('.doc', 151), ('.txt', 139), ('.rar', 112)]

    result: list[Download | None] = process_map(
        download,
        list(id_to_img_path.items()),
        chunksize=50,
        max_workers=16,
    )
    df = pd.DataFrame(list(filter(None, result)))
    df.to_csv(REPO_DATA_ROOT / "vessel-archive.csv", index=False)


if __name__ == "__main__":
    if not SCAV_PORT_ROOT.exists():
        build_8_class_folder()
        build_inlier_outlier_folder()
        build_operations_folder()

    local_files = [
        file
        for folder in [SCAV_PORT_ROOT, VESSEL_ARCHIVE_ROOT]
        for file in folder.rglob("*.jpg")
    ]

    process_map(
        s3_upload,
        local_files,
        chunksize=100,
        max_workers=8,
    )
