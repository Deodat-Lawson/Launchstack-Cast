"""
Whisper transcription worker.

Input:
  { "inputs": { "audio_path": "/path/to/audio.wav", "language": "en" } }

Output:
  { "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}], "language": "en" }

Runs as a long-lived daemon by default so the Whisper model is loaded once.
Pass --once for one-shot CLI-style execution. The model size is fixed at init
(env CAST_WHISPER_MODEL, default "base"); per-request options.model is honored
only in one-shot mode.
"""

import sys
import os

# Allow importing common module from the python/ root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.run import run, run_daemon


def init_whisper():
    import whisper

    model_name = os.environ.get("CAST_WHISPER_MODEL", "base")
    return whisper.load_model(model_name)


def transcribe(request, model=None):
    inputs = request.get("inputs", {})
    audio_path = inputs.get("audio_path")
    language = inputs.get("language")

    if not audio_path:
        raise ValueError("inputs.audio_path is required")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    if model is None:
        # One-shot path.
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
    if "--once" in sys.argv:
        run(lambda req: transcribe(req, None))
    else:
        run_daemon(transcribe, init=init_whisper)
