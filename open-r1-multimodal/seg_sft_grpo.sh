#!/usr/bin/env bash
set -euo pipefail

# ─── GPUs (Adjust as needed) ────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ─── Paths & Names (Configure Here) ─────────────────────────────────────────
PROJECT_ROOT="JUDO/open-r1-multimodal"

# Image roots for each step
IMAGE_ROOT_SEG_SFT="JUDO"
IMAGE_ROOT_DOMAIN_SFT="JUDO/MMAD"
IMAGE_ROOT_GRPO="${IMAGE_ROOT_DOMAIN_SFT}"

# Define run names and output directories for all stages
SEG_SFT_RUN_NAME="SFT_SEG"
SEG_SFT_OUT="${PROJECT_ROOT}/output/${SEG_SFT_RUN_NAME}"

DOMAIN_SFT_RUN_NAME="SFT_SEG_DOMINJ"
DOMAIN_SFT_OUT="${PROJECT_ROOT}/output/${DOMAIN_SFT_RUN_NAME}"

GRPO_RUN_NAME="JUDO_7B"
GRPO_OUT="${PROJECT_ROOT}/output/${GRPO_RUN_NAME}"


cd "${PROJECT_ROOT}"

# ─── 1) Run Segmentation SFT ────────────────────────────────────────────────
echo "=== [1/6] Running Segmentation SFT → ${SEG_SFT_OUT} ==="
accelerate launch \
  --config_file configs/zero3.yaml \
  --num_processes 4 \
  src/open_r1/seg_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name data_config/sft_seg.yaml \
    --image_root "${IMAGE_ROOT_SEG_SFT}" \
    --grid_size 16 \
    --coverage_thresh 0.02 \
    --learning_rate 1e-6 \
    --num_train_epochs 8 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 20 \
    --run_name "${SEG_SFT_RUN_NAME}" \
    --output_dir "${SEG_SFT_OUT}" \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1

# ─── 2) Patch Segmentation SFT generation_config.json ───────────────────────
echo "=== [2/6] Patching SEG SFT generation_config.json → transformers_version=4.49.0.dev0 ==="
SFT_GEN_CONF="${SEG_SFT_OUT}/generation_config.json"
if [ -f "${SFT_GEN_CONF}" ]; then
  jq '.transformers_version = "4.49.0.dev0"' \
    "${SFT_GEN_CONF}" > "${SFT_GEN_CONF}.tmp" \
    && mv "${SFT_GEN_CONF}.tmp" "${SFT_GEN_CONF}"
  echo "→ Patched ${SFT_GEN_CONF}"
else
  echo "⚠️  No generation_config.json at ${SFT_GEN_CONF}"
fi

# ─── 3) Run Domain Injection SFT ──────────────────────────────────────────
echo "=== [3/6] Running Domain Injection SFT (from ${SEG_SFT_OUT}) → ${DOMAIN_SFT_OUT} ==="
accelerate launch \
  --config_file configs/zero3.yaml \
  --num_processes 4 \
  src/open_r1/sft.py \
    --model_name_or_path "${SEG_SFT_OUT}" \
    --dataset_name data_config/sft.yaml \
    --learning_rate 5e-7 \
    --num_train_epochs 2 \
    --packing \
    --max_seq_length 4096 \
    --image_root "${IMAGE_ROOT_DOMAIN_SFT}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 1 \
    --run_name "${DOMAIN_SFT_RUN_NAME}" \
    --output_dir "${DOMAIN_SFT_OUT}" \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1
echo "=== Domain Injection SFT complete ==="

# ─── 4) Patch Domain Injection SFT generation_config.json ───────────────────
echo "=== [4/6] Patching DOMAIN SFT generation_config.json → transformers_version=4.49.0.dev0 ==="
DOMAIN_SFT_GEN_CONF="${DOMAIN_SFT_OUT}/generation_config.json"
if [ -f "${DOMAIN_SFT_GEN_CONF}" ]; then
  jq '.transformers_version = "4.49.0.dev0"' \
    "${DOMAIN_SFT_GEN_CONF}" > "${DOMAIN_SFT_GEN_CONF}.tmp" \
    && mv "${DOMAIN_SFT_GEN_CONF}.tmp" "${DOMAIN_SFT_GEN_CONF}"
  echo "→ Patched ${DOMAIN_SFT_GEN_CONF}"
else
  echo "⚠️  No generation_config.json at ${DOMAIN_SFT_GEN_CONF}"
fi

# ─── 5) Find a free port for torchrun ───────────────────────────────────────
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "=== [5/6] Using dynamic master port: $MASTER_PORT ==="

# ─── 6) Run GRPO using the final SFT model as a base ────────────────────────
echo "=== [6/6] Running GRPO (from ${DOMAIN_SFT_OUT}) → ${GRPO_OUT} ==="
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="$MASTER_PORT" \
  src/open_r1/grpo_seg_twoimg.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${GRPO_OUT}" \
    --model_name_or_path "${DOMAIN_SFT_OUT}" \
    --dataset_name data_config/ad_seg_twoimg.yaml \
    --image_root "${IMAGE_ROOT_GRPO}" \
    --max_prompt_length 1024 \
    --num_generations 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 14 \
    --run_name "${GRPO_RUN_NAME}" \
    --save_steps 500 \
    --save_total_limit 4
echo "=== GRPO complete ==="


# ─── 7) Patch GRPO generation_config.json (optional but good practice) ──────
echo "=== [7/8] Patching GRPO generation_config.json → transformers_version=4.49.0.dev0 ==="
GRPO_GEN_CONF="${GRPO_OUT}/generation_config.json"
if [ -f "${GRPO_GEN_CONF}" ]; then
  jq '.transformers_version = "4.49.0.dev0"' \
    "${GRPO_GEN_CONF}" > "${GRPO_GEN_CONF}.tmp" \
    && mv "${GRPO_GEN_CONF}.tmp" "${GRPO_GEN_CONF}"
  echo "→ Patched ${GRPO_GEN_CONF}"
else
  echo "⚠️  No generation_config.json at ${GRPO_GEN_CONF}"
fi
