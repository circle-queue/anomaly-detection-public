#!/bin/sh

TESTING="--trial-overrides={'n_samples':32}"
BATCH_SIZE=128

MODEL1="resnet18_self-sup-con"
MODEL2="resnet18_sup-con"
MODEL3="resnet18_T-S"

TASK1="add-task--self-supervised-contrast-finetune"
# TASK2="add-task--supervised-contrast-finetune"
# TASK3="add-task--teacher-student-finetune"

PARAMS1="--params-cls=SupConParams"
# PARAMS2="--params-cls=SupConParams"
# PARAMS3="--params-cls=TSBackBoneParams"

COMMON=" \
    dataloader-vesselarchive \
        --transform=resnet18 \
        --batch-size=$BATCH_SIZE \
    add-model \
        --weights=random \
"
OPTUNA_TASK=" \
    run-optuna \
        --optuna-direction=minimize \
        --optuna-metric-name=eval_loss \
"
CHECKPOINT=" \
    checkpoint-evaluations \
    checkpoint-model \
    run \
"

# Hyperparameter search
python -m anomaly_detection $COMMON --model-name=$MODEL1 $TASK1 --cfg='"optuna"' $OPTUNA_TASK $PARAMS1 # WIP Yellow
# python -m anomaly_detection $COMMON --model-name=$MODEL2 $TASK2 --cfg="'optuna'" $OPTUNA_TASK $PARAMS2 # WIP Green
# python -m anomaly_detection $COMMON --model-name=$MODEL3 $TASK3 --cfg='"optuna"' $OPTUNA_TASK $PARAMS3 # WIP Home

# Train models with best hyperparams
python -m anomaly_detection $COMMON --model-name=$MODEL1 $TASK1 \
        --cfg="('embed_loss__vesselarchive__$(echo $MODEL1)__random','minimize','eval_loss')" \
    $CHECKPOINT

# python -m anomaly_detection $COMMON --model-name=$MODEL2 $TASK2 \
#         --cfg="('embed_loss__vesselarchive__$(echo $MODEL2)__imagenet','minimize','eval_loss')" \
#     $CHECKPOINT

# python -m anomaly_detection $COMMON --model-name=$MODEL3 $TASK3 \
#         --cfg="('embed_loss__vesselarchive__$(echo $MODEL3)__imagenet','minimize','eval_loss')" \
#     $CHECKPOINT
