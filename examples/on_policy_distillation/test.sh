#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 -f "sglang.launch_server" || true
pkill -9 -f "python -m sglang" || true

set -ex


# Start the teacher model server
TEACHER_IP="127.0.0.1" # Use localhost here, you can change it to your IP
# TEACHER_IP="10.144.205.69"
TEACHER_PORT=13141
# TEACHER_PORT=8000
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

## Launch the teacher model server in the background
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
    --model-path /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 2 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &

echo "Starting teacher model server..."

## Wait for the teacher model server to be ready
# until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    sleep 5
done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONBUFFERED=16
export WANDB_API_KEY="522a32e0a2b1b6781aabe86e432e96c99f5ca4f7"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/slime/scripts/models/qwen3-8B.sh"


CKPT_ARGS=(
   --hf-checkpoint /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B-base
   --ref-load /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B-base
   --load /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B-base
   --save /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/slime_opd
   --save-interval 50
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
#    --num-rollout 300
   --num-epoch 2
   --rollout-batch-size 3
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 12
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data aime /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/aime24/aime-2024.jsonl
#    --n-samples-per-eval-prompt 4
#    --eval-max-response-len 16384
#    --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
#    --log-probs-chunk-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime
   --wandb-group qwen3-4B-opd-test
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes 1
   --actor-num-gpus-per-node 6
   --colocate
   --rollout-num-gpus 6
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}












