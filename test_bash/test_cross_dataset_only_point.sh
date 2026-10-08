#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/test_cross_dataset_only_point_dowel.sh"
bash "${script_dir}/test_cross_dataset_only_point_cookie.sh"
bash "${script_dir}/test_cross_dataset_only_point_carrot.sh"
