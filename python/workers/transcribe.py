"""
Whisper transcription worker.

Input:
  { "inputs": { "audio_path": "/path/to/audio.wav", "language": "en" } }

Output:
  { "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}], "language": "en" }
"""

import sys
import os

# Allow importing common module from the python/ root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.run import run


def transcribe(request):
    inputs = request.get("inputs", {})
    audio_path = inputs.get("audio_path")
    language = inputs.get("language")

    if not audio_path:
        raise ValueError("inputs.audio_path is required")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    import whisper

    model_name = request.get("options", {}).get("model", "base")
    model = whisper.load_model(model_name)

    decode_opts = {}
    if language:
        decode_opts["language"] = language

    result = model.transcribe(audio_path, **decode_opts)

    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "segments": segments,
        "language": result.get("language", language or "unknown"),
    }


if __name__ == "__main__":
    run(transcribe)
