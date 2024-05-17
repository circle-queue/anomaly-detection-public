#!/bin/sh

# This file trains models using different backbones and classification methods, and evaluates it on the condition good/anomaly dataset
EVAL_DS_BASE="condition"

TESTING="--trial-overrides={'n_samples':32}"
BATCH_SIZE=1

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

TS_SAMPLES='{"n_samples":lambda trial:trial.suggest_int("n_samples",1,1000,log=True)}'
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
DS_2_NAME="lock-condition"
DS_2=" \
    dataloader-$EVAL_DS_BASE \
        --sub-dataset=$DS_2_NAME \
        --transform=resnet18 \
        --train-classes=['good'] \
        --test-classes=['good','coating-missing-or-peeled-off','broken','burn-mark','visible-cracks'] \
        --batch-size=$BATCH_SIZE \
    set-outliers \
        --labels-inliers=['good'] \
        --labels-outliers=['coating-missing-or-peeled-off','broken','burn-mark','visible-cracks'] \
    add-model \
"
PC_SAMPLES_DS_2='{"n_samples":lambda trial:20}'
DS_3_NAME="ring-condition"
DS_3=" \
    dataloader-$EVAL_DS_BASE \
        --sub-dataset=$DS_3_NAME \
        --transform=resnet18 \
        --train-classes=['good'] \
        --test-classes=['good','broken','missing','collapsed'] \
        --batch-size=$BATCH_SIZE \
    set-outliers \
        --labels-inliers=['good'] \
        --labels-outliers=['broken','missing','collapsed'] \
    add-model \
"
PC_SAMPLES_DS_3='{"n_samples":lambda trial:1}'
DS_4_NAME="ring-surface-condition"
DS_4=" \
    dataloader-$EVAL_DS_BASE \
        --sub-dataset=$DS_4_NAME \
        --transform=resnet18 \
        --train-classes=['good'] \
        --test-classes=['good','coating-cracks','coating-peel-off','scuffing','signs-of-abrasive-wear','embedded-iron','signs-of-adhesive-wear'] \
        --batch-size=$BATCH_SIZE \
    set-outliers \
        --labels-inliers=['good'] \
        --labels-outliers=['coating-cracks','coating-peel-off','scuffing','signs-of-abrasive-wear','embedded-iron','signs-of-adhesive-wear'] \
    add-model \
"
PC_SAMPLES_DS_4='{"n_samples":lambda trial:5}'

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


# Each of the following categories first performs a hyperparameter search, after which it trains & stores a model with the best hyperparams
# Classifier method=Reverse Distillation
    # ImageNet Baseline
        # Hyperparameter search
        python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
        python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
        python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
        python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
        # Train models with best hyperparams
        python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        # Test best models
        python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_BASE $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_BASE)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=Teacher-Student
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=self-supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_1_VA $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_1_VA)','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$TS_SAMPLES" $PARAMS_TS

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_RN_2_SP $TASK_TS --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_RN_2_SP)','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            # Impossible, no labels

# Classifier method=PatchCore
    # ImageNet Baseline
        python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)" $PARAMS_PC # DONE
        python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
        python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
        python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE

        python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

        python -m anomaly_detection test $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        python -m anomaly_detection test $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_BASE $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_BASE)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=Teacher-Student
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # WIP Green (20:12 03-04-2024)
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # WIP Green (20:12 03-04-2024)
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # WIP Green (20:12 03-04-2024)
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # WIP Green (20:12 03-04-2024)

            python -m anomaly_detection $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_TS --weights=$WEIGHT_TS_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_TS)__$(echo $WEIGHT_TS_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=self-supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE

            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
        # Backbone dataset=Vessel Archive
            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC

            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_1_VA $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_1_VA)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

    # Backbone method=supervised contrastive learning
        # Backbone dataset=Scavenge port
            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_1)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_2)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_3)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg='"optuna"' $OPTUNA_TASK --trial-overrides="$(echo $PC_SAMPLES_DS_4)|$(echo $PC_SAMPLE_PCT)"  $PARAMS_PC # DONE

            python -m anomaly_detection $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT

            python -m anomaly_detection test $DS_1 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_1_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_2 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_2_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_3 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_3_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
            python -m anomaly_detection test $DS_4 --model-name=$MODEL_RN --weights=$WEIGHT_RN_2_SP $TASK_PC --cfg="('anomaly__$(echo $EVAL_DS_BASE)-$(echo $DS_4_NAME)__$(echo $MODEL_RN)__$(echo $WEIGHT_RN_2_SP)_patchcore','maximize','anomaly_auc')" $CHECKPOINT
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

