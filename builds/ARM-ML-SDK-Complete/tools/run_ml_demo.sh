#!/bin/bash
# Run ML inference demo

MODEL_PATH="$1"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model.tflite>"
    exit 1
fi

echo "=== ARM ML SDK Inference Demo ==="
echo "Model: $MODEL_PATH"

# Generate scenario from model
python3 tools/create_ml_pipeline.py \
    --model "$MODEL_PATH" \
    --output scenarios/ml_inference.json

# Run inference
export DYLD_LIBRARY_PATH=/usr/local/lib
bin/scenario-runner \
    --scenario scenarios/ml_inference.json \
    --output results/

echo "Inference complete. Results in results/"
