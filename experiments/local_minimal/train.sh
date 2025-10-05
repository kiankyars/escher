#!/bin/bash
# Training script for ExoMinerSmall

set -e

# Setup Python environment
# Uncomment and modify based on your setup:
# conda activate exominer_env
# or
# source venv/bin/activate

cd /Users/kian/Code/escher

echo "=========================================="
echo "Starting ExoMinerSmall training"
echo "=========================================="

# Check if TensorFlow is available
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" || {
    echo "ERROR: TensorFlow not found. Please install dependencies first."
    exit 1
}

# Run training
python src/train/train_model.py \
    --config experiments/local_minimal/config_train.yaml \
    --mixed_precision

echo "=========================================="
echo "Training complete!"
echo "Check results in: experiments/local_minimal/run_001/"
echo "=========================================="
