#!/usr/bin/env python3
"""Test script for the fine-tuned Qwen3-TTS model."""
import argparse
import torch
import soundfile as sf
import onnxruntime
onnxruntime.set_default_logger_severity(3)  # Suppress GPU device discovery warnings

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
        default="Здравей! Това е тест на финтюнирания модел за български език.",
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
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Audio saved to: {args.output}")
    print(f"Sample rate: {sr} Hz")


if __name__ == "__main__":
    main()
