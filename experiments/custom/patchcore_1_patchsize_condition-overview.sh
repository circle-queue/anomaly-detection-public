#!/bin/sh

# This file trains models using different backbones and classification methods, and evaluates it on the condition good/anomaly dataset
EVAL_DS_BASE="condition"

TESTING="--trial-overrides={'n_samples':32}"
BATCH_SIZE=1

# Definitions - RN:ResNet TS:TeacherStudent/ReverseDistillation PC:PatchCore SP:ScavengePort VA:VesselArchive
MODEL_RN="resnet18"
# MODEL_TS="resnet18_T-S"

# WEIGHT_BASE="imagenet"
# WEIGHT_TS_SP="resnet18_T-S@scavport"
# WEIGHT_TS_VA="resnet18_T-S@vesselarchive"
# WEIGHT_RN_1_SP="resnet18_self-sup-con@scavport"
# WEIGHT_RN_1_VA="resnet18_self-sup-con@vesselarchive"
WEIGHT_RN_2_SP="resnet18_sup-con@scavport"
# WEIGHT_RN_2_VA is not possible, since VesselArchive doesnt have labels

# Anomaly classifier models
# TASK_TS="add-task--teacher-student-anomaly"
TASK_PC="add-task--patchcore-anomaly"

# PARAMS_TS="--params-cls=TSAnomalyParams"
PARAMS_PC="--params-cls=PatchCoreParams"

# TS_SAMPLES='{"n_samples":lambda trial:trial.suggest_int("n_samples",1,1000,log=True)}'
PC_SAMPLE_PCT='{"feature_retention_pct":lambda trial:trial.suggest_float("feature_retention_pct", 0.0001, 1.00, log=True)}'

# We have 4 sub-datasets
DS_1_NAME="images-scavengeport-overview"
DS_1=" \
    dataloader-$EVAL_DS_BASE \
        --sub-dataset=$DS_1_NAME \
        --transform=resnet18 \
        --train-classes=['good'] \
        --test-classes=['good','abnormal'] \
        --batch-size=$BATCH_SIZE \
    set-outliers \
        --labels-inliers=['good'] \
        --labels-outliers=['abnormal'] \
    add-model \
"
PC_SAMPLES_DS_1='{"n_samples":lambda trial:67}'
# DS_2_NAME="lock-condition"
# DS_2=" \
#     dataloader-$EVAL_DS_BASE \
#         --sub-dataset=$DS_2_NAME \
#         --transform=resnet18 \
#         --train-classes=['good'] \
#         --test-classes=['good','coating-missing-or-peeled-off','broken','burn-mark','visible-cracks'] \
#         --batch-size=$BATCH_SIZE \
#     set-outliers \
#         --labels-inliers=['good'] \
#         --labels-outliers=['coating-missing-or-peeled-off','broken','burn-mark','visible-cracks'] \
#     add-model \
# "
# PC_SAMPLES_DS_2='{"n_samples":lambda trial:20}'
# DS_3_NAME="ring-condition"
# DS_3=" \
#     dataloader-$EVAL_DS_BASE \
#         --sub-dataset=$DS_3_NAME \
#         --transform=resnet18 \
#         --train-classes=['good'] \
#         --test-classes=['good','broken','missing','collapsed'] \
#         --batch-size=$BATCH_SIZE \
#     set-outliers \
#         --labels-inliers=['good'] \
#         --labels-outliers=['broken','missing','collapsed'] \
#     add-model \
# "
# PC_SAMPLES_DS_3='{"n_samples":lambda trial:1}'
# DS_4_NAME="ring-surface-condition"
# DS_4=" \
#     dataloader-$EVAL_DS_BASE \
#         --sub-dataset=$DS_4_NAME \
#         --transform=resnet18 \
#         --train-classes=['good'] \
#         --test-classes=['good','coating-cracks','coating-peel-off','scuffing','signs-of-abrasive-wear','embedded-iron','signs-of-adhesive-wear'] \
#         --batch-size=$BATCH_SIZE \
#     set-outliers \
#         --labels-inliers=['good'] \
#         --labels-outliers=['coating-cracks','coating-peel-off','scuffing','signs-of-abrasive-wear','embedded-iron','signs-of-adhesive-wear'] \
#     add-model \
# "
# PC_SAMPLES_DS_4='{"n_samples":lambda trial:5}'

OPTUNA_TASK=" \
    run-optuna \
        --optuna-direction=maximize \
        --optuna-metric-name=anomaly_auc \
"
CHECKPOINT=" \
    checkpoint-evaluations \
    checkpoint-model \
    run \
"

CFG="\
    anomaly_scorer_num_nn=4, \
    feature_retention_pct=0.0003146061763535602, \
    imagesize=(3, 224, 224), \
    layers_to_extract_from='layer3', \
    n_samples=67, \
    pretrain_embed_dimension=192, \
    target_embed_dimension=256, \
"
    # patchsize=12, \

CHECKPOINT=" \
    checkpoint-evaluations \
    checkpoint-model \
    run \
"

python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC "--cfg=dict($CFG pixel_agg_func='mean', patchsize=1)" prefix-experiment ablation-patch-mean-patchsize1_ $CHECKPOINT