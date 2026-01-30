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
    
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines_json = f.readlines()
        
        valid_lines = []
        for line in lines_json:
            try:
                entry = json.loads(line)
                if Path(entry["audio"]).exists():
                    valid_lines.append(line)
            except:
                pass
        
        if len(valid_lines) < len(lines_json):
            print(f"cleaned {len(lines_json) - len(valid_lines)} invalid entries from {jsonl_path}")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)
            existing_entries = len(valid_lines)
        else:
             existing_entries = len(lines_json)
    else:
        existing_entries = 0
    
    print(f"Resuming from {existing_entries} sentences.")
    
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
                retry_count = 0
                max_retries = 5
                wait_time = 10
                
                while retry_count < max_retries:
                    saved_path = generate_audio(text, str(file_path))
                    if saved_path:
                        break
                    
                    print(f"Failed to generate audio for line {i}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    wait_time *= 2 # Exponential backoff

                if not file_path.exists():
                    print(f"Skipping line {i} after max retries.")
                    continue
                
                # If this is the first successful generation and we don't have a ref audio, copy it
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
