import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from gemini import generate_audio


sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Qwen3-TTS")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sentences to process (0 for all)")
    args = parser.parse_args()

    sentences_file = Path("src/data/sentences.txt")
    output_dir = Path("src/data/audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = Path("src/data/train.jsonl")
    
    with open(sentences_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} sentences.")
    
    ref_audio_path = output_dir / "ref_audio.wav"
    
    existing_entries = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            existing_entries = len(f.readlines())
    
    print(f"Skipping first {existing_entries} sentences.")
    
    count = 0
    with open(jsonl_path, "a", encoding="utf-8") as f_json:
        for i, line in enumerate(lines):
            if i < existing_entries:
                continue
            
            if args.limit > 0 and count >= args.limit:
                print(f"Reached limit of {args.limit} sentences.")
                break

            text = line.strip()
            if not text:
                continue
            
            filename = f"utt{i:05d}.wav"
            file_path = output_dir / filename
            
            print(f"Generating [{i}]: {text[:30]}...")
            
            if not file_path.exists():
                saved_path = generate_audio(text, str(file_path))
                if not saved_path:
                    print(f"Failed to generate audio for line {i}")
                    time.sleep(5)
                    continue
                
                if not ref_audio_path.exists():
                    shutil.copy(saved_path, ref_audio_path)
                    print(f"Set reference audio to {ref_audio_path}")
            
            entry = {
                "audio": str(file_path.absolute()),
                "text": text,
                "ref_audio": str(ref_audio_path.absolute())
            }
            f_json.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_json.flush()
            
            count += 1
            time.sleep(2) 

if __name__ == "__main__":
    main()
