import os
import time
import base64
import struct
import mimetypes
from google import genai
from pathlib import Path
from google.genai import types
from dotenv import load_dotenv


env_path = Path(__file__).parent / "configs" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")

def generate_audio(text: str, output_path: str, voice_name: str = "Puck") -> str | None:
    """
    Generates audio for the given text using Gemini TTS and saves it to output_path.

    Args:
        text: The text to convert to speech.
        output_path: The path to save the generated audio file.
        voice_name: The voice to use (default: "Puck").

    Returns:
        The path to the saved file if successful, None otherwise.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return None

    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-pro-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    try:
        saved_file = False
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                inline_data = part.inline_data
                data_buffer = inline_data.data
                
                if not Path(output_path).suffix:
                     pass

                if not output_path.endswith(".wav"):
                     output_path += ".wav"
                
                mime_type = inline_data.mime_type or "audio/wav"
                ext = mimetypes.guess_extension(mime_type)
                
                if ext is None or mime_type == "audio/pcm" or "rate=" in mime_type:
                     data_buffer = convert_to_wav(inline_data.data, mime_type)
                
                save_binary_file(output_path, data_buffer)
                saved_file = True
                break 
        
        if saved_file:
            return output_path
        else:
            print("No audio data received from API.")
            return None

    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """
    Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    test_text = "Това е тестов запис за проверка на системата."
    output = generate_audio(test_text, "test_audio.wav")
    if output:
        print(f"Successfully generated: {output}")
    else:
        print("Failed to generate audio.")
