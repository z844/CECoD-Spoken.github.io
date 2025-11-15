#!/usr/bin/env bash
set -e

WAV_FILE=""  #data/input/###.wav
MODEL_ID="model/Llama3-8B-Chinese-Chat"
LORA_WEIGHT="model/CECoD_lora"
CUDA_NUM="cuda:0"

# 路径（脚本路径）
SECAP_SCRIPT="sh1.py"
MAIN_SCRIPT="sh2.py"

conda run -n secap python "${SECAP_SCRIPT}" \
    --audio "${WAV_FILE}" \
    --cuda 0

conda run -n cosyvoice "${MAIN_SCRIPT}" \
    --wav_file "${WAV_FILE}" \
    --cuda_num "${CUDA_NUM}" \
    --model_id "${MODEL_ID}" \
    --lora_weight "${LORA_WEIGHT}"

