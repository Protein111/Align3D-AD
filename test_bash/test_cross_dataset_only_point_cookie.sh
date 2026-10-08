#!/usr/bin/env bash
set -e

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

data_root="${DATA_ROOT:-${project_root}/data}"
REAL3D_DATA_PATH="${REAL3D_DATA_PATH:-${data_root}/Real3D-AD}"

device="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES-0}}"
exp_root="${EXP_ROOT:-${project_root}/Model_weights}"
data_path="${REAL3D_DATA_PATH}"

obj_list=("cookie")
cls_ids=(0)
for cls_id in "${!cls_ids[@]}";do
    echo ${cls_id}
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            check_dir="${exp_root}/mvtec_3d/${obj_list[cls_id]}/weight/"

            CUDA_VISIBLE_DEVICES=${device} python test_only_point.py --dataset real_pc_3d_rgb  \
            --data_path "${data_path}" --save_path "./results/mvtec_to_real3d/${obj_list[cls_id]}/test" \
            --checkpoint_path "${check_dir}epoch_15.pth" \
            --image_level_weight 0 --point_level_weight 0 --need_blur False \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
        done
    done
done
