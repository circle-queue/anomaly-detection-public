# Anomaly detection

## Disclaimer
The folders under [src](src) contains *modified* version of the following repositories:
* https://github.com/pderoovere/dimo/tree/caa7190c20a54def534f83de885d6eb04da88bbd
* https://github.com/amazon-science/patchcore-inspection/tree/fcaa92f124fb1ad74a7acf56726decd4b27cbcad
* https://github.com/hq-deng/RD4AD/tree/6554076872c65f8784f6ece8cfb39ce77e1aee12
* https://github.com/HobbitLong/SupContrast/tree/66a8fe53880d6a1084b2e4e0db0a019024d6d41a

## Prerequsites
* CUDA 12.1 (Windows install: `winget install -e --id Nvidia.CUDA -v 12.1`)
* For patchcore, faiss-gpu is required, which is Linux only

## Installation
```sh
python -m pip install poetry
poetry config virtualenvs.in-project true

poetry install --with dev --with ipy --with plotting
poetry shell
```

## Setup
```sh
# These scripts must be run in seperate CMDs and kept alive while metrics are logged
poetry run mlflow server
# Allow sagemaker notebooks to connect to the mlflow server (Assumes you've configureed ~/.ssh/config, e.g. using `gdmake`)
ssh -o ExitOnForwardFailure=yes -R 5000:127.0.0.1:5000 christoffer-sommerlund-thesis
ssh -o ExitOnForwardFailure=yes -R 5000:127.0.0.1:5000 christoffer-sommerlund-patchcore
```

## Run
The entire suite of experiments are located in [experiments](experiments). The backbones must be run before the classifiers
```sh
source experiments/backbones/train_scavport_backbones.sh
source experiments/backbones/train_vesselarchive_backbones.sh
source experiments/backbones/train_backbones_for_scavport_clfs.sh
source experiments/backbones/train_backbones_for_condition_clfs.sh
```
NOTE: This will take multiple days to complete and may crash if your GPU has below 24GB ram and is not on a Linux machine.

To run a single experiment, you can invoke it from the CLI, e.g..
```sh
python -m anomaly_detection
    dataloader-condition
        --sub-dataset=images-scavengeport-overview
        --transform=resnet18
        --train-classes=['good']
        --test-classes=['good','abnormal']
        --batch-size=1
    set-outliers
        --labels-inliers=['good']
        --labels-outliers=['abnormal']
    add-model
        --model-name=resnet18_T-S
        --weights=imagenet
    add-task--teacher-student-anomaly
        --cfg='"optuna"'
    run-optuna
        --optuna-direction=maximize
        --optuna-metric-name=anomaly_auc
        --trial-overrides="{"n_samples":lambda trial:trial.suggest_int("n_samples",1,1000,log=True)}"
        --params-cls=TSAnomalyParams
```

To add a custom dataset, you need to add a command in [\_\_main\_\_.py](src\anomaly_detection\__main__.py), which invokes [model_builder.py](src\anomaly_detection\model_builder.py). The datasets are assumed to have the following structure
```.../{dataset_name}/{train,dev,test}/{label1,label2,label3}/{*.jpg}```

## Architecture
![](docs/ts_architecture.png)
source: https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf
