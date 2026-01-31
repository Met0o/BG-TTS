#!/usr/bin/env python3
import argparse
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen3-TTS model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="src/output/checkpoint-epoch-2",
        help="Path to the fine-tuned checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Холдън спря, принуждавайки и другите да се спрат зад него.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="puck_bulgarian",
        help="Speaker name (must match --speaker_name used during training)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0 or cpu)",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print(f"Generating speech for: '{args.text}'")
    
    # Prepend language tag to match fine-tuning
    input_text = args.text
    if not input_text.strip().startswith("(Bulgarian)"):
        input_text = f"(Bulgarian) {input_text}"
        
    wavs, sr = tts.generate_custom_voice(
        text=input_text,
        speaker=args.speaker,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Audio saved to: {args.output}")
    print(f"Sample rate: {sr} Hz")


if __name__ == "__main__":
    main()
