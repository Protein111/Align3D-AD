#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/test_mvtec_dowel.sh"
bash "${script_dir}/test_mvtec_cookie.sh"
bash "${script_dir}/test_mvtec_carrot.sh"
