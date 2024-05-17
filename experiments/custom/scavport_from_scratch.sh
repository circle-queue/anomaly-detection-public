#!/bin/sh

TESTING="--trial-overrides={'n_samples':32}"
BATCH_SIZE=128

# MODEL1="resnet18_self-sup-con"
MODEL2="resnet18_sup-con"
# MODEL3="resnet18_T-S"

# TASK1="add-task--self-supervised-contrast-finetune"
TASK2="add-task--supervised-contrast-finetune"
# TASK3="add-task--teacher-student-finetune"

# PARAMS1="--params-cls=SupConParams"
PARAMS2="--params-cls=SupConParams"
# PARAMS3="--params-cls=TSBackBoneParams"

COMMON=" \
    dataloader-scavport \
        --transform=resnet18 \
        --train-classes=['liner','piston-ring-overview','topland','piston-top','single-piston-ring','skirt','scavange-box','piston-rod'] \
        --test-classes=['liner','piston-ring-overview','topland','piston-top','single-piston-ring','skirt','scavange-box','piston-rod'] \
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
# python -m anomaly_detection $COMMON --model-name=$MODEL1 $TASK1 --cfg='"optuna"' $OPTUNA_TASK $PARAMS1
python -m anomaly_detection $COMMON --model-name=$MODEL2 $TASK2 --cfg="'optuna'" $OPTUNA_TASK $PARAMS2
# python -m anomaly_detection $COMMON --model-name=$MODEL3 $TASK3 --cfg='"optuna"' $OPTUNA_TASK $PARAMS3

# Train models with best hyperparams
# python -m anomaly_detection $COMMON --model-name=$MODEL1 $TASK1 \
#         --cfg="('embed_loss__scavport__$(echo $MODEL1)__imagenet','minimize','eval_loss')" \
#     $CHECKPOINT

python -m anomaly_detection $COMMON --model-name=$MODEL2 $TASK2 \
        --cfg="('embed_loss__scavport__$(echo $MODEL2)__random','minimize','eval_loss')" \
    $CHECKPOINT

# python -m anomaly_detection $COMMON --model-name=$MODEL3 $TASK3 \
#         --cfg="('embed_loss__scavport__$(echo $MODEL3)__imagenet','minimize','eval_loss')" \
#     $CHECKPOINT