set -euo pipefail
torchrun --nproc_per_node=8 -m physmel_lv_world.gateways.train settings/longhaul/main.yaml
