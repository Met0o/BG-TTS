# BG-TTS (Bulgarian Text-to-Speech)

This project aims to fine-tune the **Qwen3-TTS** model for Bulgarian speech synthesis. We use **Google Gemini TTS** (via the `gemini-2.5-pro-preview-tts` model and "Puck" voice) to generate a high-quality synthetic dataset from Bulgarian text.

## Project Structure

- `src/gemini.py`: Core script for interacting with the Google Gemini API for TTS.
- `src/generate_dataset.py`: Automates dataset generation by reading sentences from `src/data/sentences.txt` and creating WAV files and a JSONL manifest.
- `src/setup_finetuning.sh`: Sets up the Qwen3-TTS repository and installs dependencies.
- `src/finetune.sh`: Runs the data preparation and fine-tuning process using Qwen3-TTS scripts.
- `src/data/sentences.txt`: Source text for the dataset.
- `src/data/audio/`: Directory where generated audio files are stored.
- `src/Qwen3-TTS/`: Cloned repository of Qwen3-TTS (after setup).

## Prerequisites

1.  **NVIDIA GPU**: Required for fine-tuning (tested on RTX 3090 / A4000).
2.  **Google API Key**: You need a valid Google API key with access to the Gemini API.
    -   Create a file `src/configs/.env` and add:
        ```bash
        GOOGLE_API_KEY=your_api_key_here
        ```

## Usage

### 1. Generate Dataset

First, generate the synthetic audio dataset. This script reads sentences from `src/data/sentences.txt`, generates audio using Gemini, and creates the `src/data/train.jsonl` manifest required for Qwen3-TTS.

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate dataset (use --limit for testing)
python src/generate_dataset.py --limit 100
# To generate the full dataset:
# python src/generate_dataset.py
```

This process will create:
- `src/data/audio/*.wav`: The audio files.
- `src/data/train.jsonl`: The training manifest.
- `src/data/audio/ref_audio.wav`: A reference audio file for speaker consistency.

### 2. Setup Fine-Tuning Environment

Run the setup script to clone the Qwen3-TTS repository and install the necessary Python dependencies.

```bash
chmod +x src/setup_finetuning.sh
./src/setup_finetuning.sh
```

### 3. Run Fine-Tuning

Once the dataset is ready and the environment is set up, start the fine-tuning process. This script handles data preprocessing (tokenization/code extraction) and launches the training loop.

```bash
chmod +x src/finetune.sh
./src/finetune.sh
```

The script uses the following default configuration (adjustable in `src/finetune.sh`):
- **Base Model**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- **Batch Size**: 2
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Device**: `cuda:0`

### Output

Checkpoints and logs will be saved in the `src/output/` directory.

## Notes on Dependencies

- The project uses a local `.venv`.
- `requirements.txt` contains base dependencies.
- Qwen3-TTS dependencies are installed via `pip install qwen-tts` in the setup script.

## Troubleshooting

### Rate Limits (429 Resource Exhausted)
The generation script now includes automatic retries with exponential backoff. If you still encounter issues, try running with a smaller limit or wait for your quota to reset.
```bash
python src/generate_dataset.py --limit 50
```

### Missing Audio Files / File Not Found
If you interrupted the generation or deleted files, `train.jsonl` might be out of sync.
**Fix**: Delete `src/data/train.jsonl` and run the generation script again. It will skip existing audio files and regenerate the manifest.

### SoX Error / librosa / soundfile
If you see errors like `SoX could not be found` or `libsndfile` issues:
1.  Install system dependencies:
    ```bash
    sudo apt-get install sox libsndfile1 ffmpeg
    ```
2.  If using `finetune.sh`, ensure you have these installed.
