cd /home/ec2-user/SageMaker/anomaly-detection/src/patchcore-inspection/
datapath=/home/ec2-user/SageMaker/anomaly-detection/src/patchcore-inspection/condition-ds
datasets=('lock-condition' 'ring-condition' 'ring-surface-condition' 'images-scavengeport-overview')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
# wideresnet50
# python -W ignore bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project condition-ds_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" condition-ds $datapath

# wideresnet50 10%
python -W ignore bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P05_D1024-1024_PS-3_AN-1_S0 --log_project condition-ds_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.5 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" condition-ds $datapath


# python -W ignore bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_vitSWINbase_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project condition-ds_Results results \
# patch_core -b vit_swin_base -le layer1 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 3 --patchsize 5 sampler -p 0.50 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" condition-ds $datapath


