#!/usr/bin/env bash
set -e

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

data_root="${DATA_ROOT:-${project_root}/data}"
MVTEC_DATA_PATH="${MVTEC_DATA_PATH:-${data_root}/MVTec3D-AD_x_9_views}"

device="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES-0}}"
seeds=(111)
exp_root="${EXP_ROOT:-${project_root}/Model_weights}"

alignment_loss_scales=(1)
lambdas=(0.1)

obj_list=("dowel")
cls_ids=(0)
for cls_id in "${!cls_ids[@]}";do
    echo ${cls_id}
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            for seed in "${seeds[@]}";do
                for alignment_loss_scale in "${alignment_loss_scales[@]}";do
                    for lambda in "${lambdas[@]}";do
                        save_dir="${exp_root}/mvtec_3d/${obj_list[cls_id]}/weight/"
                        LOG=${save_dir}res.log
                        echo "${LOG}"

                        CUDA_VISIBLE_DEVICES=${device} python train_v12.py --dataset mvtec_pc_3d_rgb --train_data_path "${MVTEC_DATA_PATH}" \
                        --save_path "${save_dir}" \
                        --features_list 24 --image_size 336  --batch_size 4 --print_freq 1 \
                        --epoch_1 250 --epoch_2 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_dataset_name ${obj_list[cls_id]} --seed ${seed} \
                        --alignment_loss_scale ${alignment_loss_scale} --align_local_weight_lambda ${lambda}

                        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec_pc_3d_rgb \
                        --data_path "${MVTEC_DATA_PATH}" --save_path "./results/mvtec_3d/${obj_list[cls_id]}/test" \
                        --checkpoint_path "${save_dir}epoch_15.pth" \
                        --image_level_weight 0 --point_level_weight 0 --need_blur False \
                        --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
                    done
                done
            wait
            done
        done
    done
done
