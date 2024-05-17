#!/bin/sh

# This file trains models using different backbones and classification methods, and evaluates it on the scavenge port in-/outlier dataset
EVAL_DS="scavport"

TESTING="--trial-overrides={'n_samples':32}"
BATCH_SIZE=16

# Definitions - RN:ResNet TS:TeacherStudent/ReverseDistillation PC:PatchCore SP:ScavengePort VA:VesselArchive
MODEL_RN="resnet18"
MODEL_TS="resnet18_T-S"

WEIGHT_BASE="imagenet"
WEIGHT_TS_SP="resnet18_T-S@scavport"
WEIGHT_TS_VA="resnet18_T-S@vesselarchive"
WEIGHT_RN_1_SP="resnet18_self-sup-con@scavport"
WEIGHT_RN_1_VA="resnet18_self-sup-con@vesselarchive"
WEIGHT_RN_2_SP="resnet18_sup-con@scavport"
# WEIGHT_RN_2_VA is not possible, since VesselArchive doesnt have labels

# Anomaly classifier models
TASK_TS="add-task--teacher-student-anomaly"
TASK_PC="add-task--patchcore-anomaly"

PARAMS_TS="--params-cls=TSAnomalyParams"
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
"
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

SCAVPORT_PC_SAMPLES='{"n_samples":lambda trial:trial.suggest_int("n_samples",10,4521,log=True)}'

# Each of the following categories first performs a hyperparameter search, after which it trains & stores a model with the best hyperparams
# Classifier method=Reverse Distillation
    # ImageNet Baseline
        # Hyperparameter search
        python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
        # Train models with best hyperparams
        python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        # saved models
        python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=Teacher-Student
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=self-supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK $PARAMS_TS
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            # Impossible, no labels

# Classifier method=PatchCore
    # ImageNet Baseline
        python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC # DONE
        python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=Teacher-Student
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC # DONE
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC # WIP Green (20:12 03-04-2024)
            python -m anomaly_detection $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=self-supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC # DONE
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$SCAVPORT_PC_SAMPLES" $PARAMS_PC # DONE
            python -m anomaly_detection $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $COMMON --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            # Impossible, no labels


# # Train model with custom hyperparams
# # python -m anomaly_detection $COMMON --model-name=$MODEL1 $TASK1
# #    add-task--teacher-student-finetune \
# #         --cfg="dict(
# #             n_samples=32,
# #             lr_bottleneck=0.001,
# #             lr_student=0.001,
# #             beta1=0.9,
# #             beta2=0.999,
# #             lr_factor_1cycle=0.01,
# #             )" \
# #   $CHECKPOINT


