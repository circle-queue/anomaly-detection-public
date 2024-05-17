cd /home/ec2-user/SageMaker/anomaly-detection/src/patchcore-inspection/
datapath=/home/ec2-user/SageMaker/anomaly-detection/src/patchcore-inspection/condition-ds
loadpath=/home/ec2-user/SageMaker/anomaly-detection/src/patchcore-inspection/results/condition-ds_Results

# modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
modelfolder=IM224_WR50_L2-3_P05_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

datasets=('lock-condition' 'ring-condition' 'ring-surface-condition' 'images-scavengeport-overview')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/condition-ds_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python -W ignore bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" condition-ds $datapath
