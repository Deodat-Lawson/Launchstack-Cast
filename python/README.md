# Cast — Python Workers

Narrow ML inference workers invoked by the Go orchestrator as subprocesses.

## Setup

```bash
cd python
uv venv
uv pip install -e ".[dev]"
```

## Running a worker manually

```bash
echo '{"request_id":"test","inputs":{"audio_path":"sample.wav"}}' | python workers/transcribe.py
```

## Contract

- **stdin**: single JSON object (see `common/run.py` for envelope schema).
- **stdout**: single JSON response object. Reserved for the protocol — do not print anything else.
- **stderr**: free-form logs and tracebacks.
- **exit 0**: success, `ok: true` in response.
- **exit 1**: failure, `ok: false` with structured error in response.
