#!/bin/bash
set -e

TARGET_DIR="src/Qwen3-TTS"

echo "Setting up Qwen3-TTS fine-tuning environment..."

if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$TARGET_DIR"
else
    echo "Qwen3-TTS repository already exists."
fi

echo "Installing Qwen3-TTS dependencies..."
./.venv/bin/pip install qwen-tts

echo "Setup complete. You can now use src/finetune.sh to start training."
