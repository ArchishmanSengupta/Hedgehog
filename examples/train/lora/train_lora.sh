#!/bin/bash
# LoRA fine-tuning example for Diffusion Language Models
# This script demonstrates fine-tuning a DLM with LoRA

set -e

# Configuration
MODEL_TYPE="dit"
DATASET="tiny-shakespeare"
OUTPUT_DIR="output/lora_dlm"

# LoRA configuration
USE_PEFT=true
PEFT_TYPE="lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training configuration
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=8
LEARNING_RATE=1e-4
MAX_SEQ_LEN=512
NUM_TIMESTEPS=1000

echo "Starting LoRA fine-tuning..."
echo "Model: $MODEL_TYPE"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"

# Run training
hedgehog train \
    --model_type $MODEL_TYPE \
    --dataset $DATASET \
    --use_peft \
    --peft_type $PEFT_TYPE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_len $MAX_SEQ_LEN \
    --num_timesteps $NUM_TIMESTEPS \
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 \
    --save_steps 100

echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
