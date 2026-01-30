import os
import openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "configs" / ".env")

speech_file_path = Path(__file__).parent / "speech.wav"

client = openai.OpenAI(
    api_key=os.getenv("API_KEY")
)

with client.audio.speech.with_streaming_response.create(
  model="gpt-4o-mini-tts-2025-12-15",
  voice="ash",
  input="Независимо от това говори се, че при даден случай той казал Полезно е да постъпваш очевидно, особено ако имаш репутацията, че си лукав."
) as response:
  response.stream_to_file(speech_file_path)
