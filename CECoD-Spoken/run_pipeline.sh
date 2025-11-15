#!/usr/bin/env bash
set -e

MODEL_ID="model/Llama3-8B-Chinese-Chat"
LORA_WEIGHT="model/CECoD_lora"
CUDA_NUM="cuda:0"

# 路径（脚本路径）
SECAP_SCRIPT="sh1.py"
MAIN_SCRIPT="sh2.py"

conda run -n secap python "${SECAP_SCRIPT}" \
    --cuda "${CUDA_NUM}"

conda run -n cosyvoice python "${MAIN_SCRIPT}" \
    --cuda_num "${CUDA_NUM}" \
    --model_id "${MODEL_ID}" \
    --lora_weight "${LORA_WEIGHT}"

