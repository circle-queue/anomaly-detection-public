#!/bin/sh

# This file trains models using different backbones and classification methods, and evaluates it on the scavenge port in-/outlier dataset
EVAL_DS="scavport"
BATCH_SIZE=16

# Definitions - RN:ResNet TS:TeacherStudent/ReverseDistillation PC:PatchCore SP:ScavengePort VA:VesselArchive
MODEL_RN="resnet18"
WEIGHT_BASE="imagenet"
PARAMS_PC="--params-cls=PatchCoreParams"

COMMON=" \
    dataloader-$EVAL_DS \
        --transform=resnet18 \
        --train-classes=['liner','piston-ring-overview','topland','piston-top','single-piston-ring','skirt','scavange-box','piston-rod'] \
        --test-classes=['inlier','outlier'] \
        --batch-size=$BATCH_SIZE \
    set-outliers \
        --labels-inliers=['inlier'] \
        --labels-outliers=['outlier'] \
    add-model \
        --model-name=$MODEL_RN \
        --weights=$WEIGHT_BASE \
    add-task--patchcore-anomaly \
"
CFG="\
    anomaly_scorer_num_nn=4, \
    feature_retention_pct=0.006028325872730335, \
    imagesize=(3, 224, 224), \
    n_samples=1671.0, \
    pretrain_embed_dimension=256.0, \
    target_embed_dimension=128.0, \
"
CHECKPOINT=" \
    checkpoint-evaluations \
    checkpoint-model \
    run \
"
python -m anomaly_detection $COMMON "--cfg=dict($CFG layers_to_extract_from='layer3', \
        pixel_agg_func='mean', patchsize=1)" prefix-experiment ablation-patch-mean-patchsize1_ $CHECKPOINT
python -m anomaly_detection $COMMON "--cfg=dict($CFG layers_to_extract_from='layer1,layer2,layer3', \
        pixel_agg_func='mean', patchsize=1)" prefix-experiment ablation-patch-mean-patchsize1-layer123_ $CHECKPOINT