#!/usr/bin/env bash
set -e

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

data_root="${DATA_ROOT:-${project_root}/data}"
EYECANDIES_DATA_PATH="${EYECANDIES_DATA_PATH:-${data_root}/Eyecandies}"

device="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES-0}}"
exp_root="${EXP_ROOT:-${project_root}/Model_weights}"

obj_list=('LicoriceSandwich')
cls_ids=(0)
for cls_id in "${!cls_ids[@]}";do
    echo ${cls_id}
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            save_dir="${exp_root}/eye_3d/${obj_list[cls_id]}/weight/"

            CUDA_VISIBLE_DEVICES=${device} python test.py --dataset eye_pc_3d_rgb  \
            --data_path "${EYECANDIES_DATA_PATH}" --save_path "./results/eye_3d/${obj_list[cls_id]}/test" \
            --checkpoint_path "${save_dir}epoch_15.pth" \
            --point_level_weight 1 \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
        done
    done
done
