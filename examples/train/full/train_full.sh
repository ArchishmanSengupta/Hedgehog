#!/bin/bash
# Full parameter fine-tuning example for Diffusion Language Models
# This script demonstrates fine-tuning all model parameters

set -e

# Configuration
MODEL_TYPE="dit"
DATASET="tiny-shakespeare"
OUTPUT_DIR="output/full_dlm"

# Training configuration
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=4
LEARNING_RATE=5e-5
MAX_SEQ_LEN=512
NUM_TIMESTEPS=1000

# Model configuration
VOCAB_SIZE=32768
HIDDEN_SIZE=384
NUM_HEADS=6
NUM_LAYERS=12
DROPOUT=0.1

echo "Starting full parameter fine-tuning..."
echo "Model: $MODEL_TYPE"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"

# Run training
hedgehog train \
    --model_type $MODEL_TYPE \
    --vocab_size $VOCAB_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --dataset $DATASET \
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
