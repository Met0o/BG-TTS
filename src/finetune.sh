#!/bin/bash
set -e

REPO_ROOT="$(pwd)"
QWEN_FINETUNE_DIR="$REPO_ROOT/src/Qwen3-TTS/finetuning"
DATA_DIR="$REPO_ROOT/src/data"
OUTPUT_DIR="$REPO_ROOT/src/output"
PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"

mkdir -p "$OUTPUT_DIR"

RAW_JSONL="$DATA_DIR/train.jsonl"
TRAIN_JSONL="$DATA_DIR/train_with_codes.jsonl"

if [ ! -d "$QWEN_FINETUNE_DIR" ]; then
    echo "Error: Qwen3-TTS fine-tuning directory not found at $QWEN_FINETUNE_DIR"
    echo "Please run src/setup_finetuning.sh first."
    exit 1
fi

INIT_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER="Qwen/Qwen3-TTS-Tokenizer-12Hz"

BATCH_SIZE=2
LR=2e-5
EPOCHS=25
SPEAKER_NAME="Puck_Bulgarian"
DEVICE="cuda:0"

echo "=== Step 1: Preparing Data (extracting audio codes) ==="
if [ ! -f "$RAW_JSONL" ]; then
    echo "Error: $RAW_JSONL not found. Please run src/generate_dataset.py first."
    exit 1
fi

echo "Running prepare_data.py..."
cd "$QWEN_FINETUNE_DIR"

"$PYTHON_EXEC" prepare_data.py \
  --device "$DEVICE" \
  --tokenizer_model_path "$TOKENIZER" \
  --input_jsonl "$RAW_JSONL" \
  --output_jsonl "$TRAIN_JSONL"

echo "=== Step 2: Running Fine-tuning (SFT) ==="

"$PYTHON_EXEC" sft_12hz.py \
  --init_model_path "$INIT_MODEL" \
  --output_model_path "$OUTPUT_DIR" \
  --train_jsonl "$TRAIN_JSONL" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --num_epochs "$EPOCHS" \
  --save_interval 5 \
  --speaker_name "$SPEAKER_NAME"

echo "=== Fine-tuning Complete ==="
echo "Checkpoints are in $OUTPUT_DIR"
