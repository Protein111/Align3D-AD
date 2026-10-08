#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/test_eyecandies_PeppermintCandy.sh"
bash "${script_dir}/test_eyecandies_LicoriceSandwich.sh"
bash "${script_dir}/test_eyecandies_Confetto.sh"
