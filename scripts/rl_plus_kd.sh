#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python




set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_API_KEY="522a32e0a2b1b6781aabe86e432e96c99f5ca4f7"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/root/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B
   --load /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B
   --save /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/slime_kd_plus_rl_test
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   # --num-rollout 3000
   --num-epoch 5
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/aime24/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

DISTILL_ARGS=(
   # Path to the teacher model for distillation
   --distill-checkpoint /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-32B
   # Coefficient for the distillation loss
   --distill-coef 0.3
   # Ratio of top entropy tokens to apply distillation loss (optional)
   --distill-top-entropy-ratio 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style cosine
   --warmup-ratio 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

if [ -z "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=()
else
    WANDB_ARGS=(
    --use-wandb
    --wandb-project slime
    --wandb-group qwen3-4B-kd_plus-rl
    --wandb-key ${WANDB_API_KEY}
    )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
   --use-fault-tolerance
   --dump-details /root/shared_data/qwen3-4B-fsdp-1116-noref/dump_details
   # --fsdp-cpu-offload
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${DISTILL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRAIN_BACKEND_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"



