# pyworker

The Go ↔ Python boundary for Cast. Workers (transcription, visual embedding,
face detection) run as Python subprocesses; this package owns the contract for
spawning them, sending requests, and reading responses.

## Two abstractions

| | `Runner` | `Pool` |
|---|---|---|
| Process lifetime | One-shot per call | Long-lived daemon, one per kind |
| Wire format | Single JSON blob via stdio | 4-byte BE length + JSON frames, looped |
| When to use | ffmpeg, scenedetect — anything where there's no model to keep warm | Whisper, CLIP, InsightFace — model load dominates per-call cost |
| Concurrency | Fresh process each call, fully parallel | Serial **per kind**; different kinds run in parallel |
| Failure mode | Process exits, error returned | Daemon evicted on IO error / timeout / crash; respawned on next call |

A typical worker process wires up both — `Runner` for the cheap stuff and `Pool` for the ML stuff. See [cmd/cast-worker/main.go](../../cmd/cast-worker/main.go).

## Wire format

Both sides share an envelope:

```json
// Request
{ "request_id": "...", "inputs": { ... }, "options": { ... } }

// Response
{ "request_id": "...", "ok": true, "data": { ... }, "error": null,
  "meta": { "duration_ms": 123 } }
```

`Runner` sends one envelope as raw JSON on stdin and reads one back on stdout. The process exits when done.

`Pool` wraps each envelope with a 4-byte big-endian length prefix and loops on the same stdin/stdout. Length-prefix framing sidesteps stdout-buffering ambiguity and means a stray `print` from a handler can't desync the protocol.

In both cases stderr is inherited so Python tracebacks land in the worker log.

## The Python side

[python/common/run.py](../../python/common/run.py) provides matching entry points:

```python
from common.run import run, run_daemon

# One-shot — pairs with Runner.
def handler(request):
    return {"result": ...}

run(handler)
```

```python
# Long-lived — pairs with Pool.
def init():
    return {"model": load_model()}   # runs once

def handler(request, state):
    return {"embeddings": state["model"].encode(...)}

run_daemon(handler, init=init)
```

A handler exception in `run_daemon` does **not** kill the process. The daemon writes an error response and keeps looping. This is intentional: a malformed input shouldn't force a 30s model reload. Process exit is reserved for unrecoverable conditions (framing errors, crashes).

## Pool semantics

- **Lazy spawn**: a daemon is created on the first `Pool.Call` for that kind and held forever.
- **Serial per kind**: one in-flight `Call` per daemon, enforced by a per-daemon mutex. Concurrent calls to *different* kinds run in parallel.
- **Eviction triggers respawn**: any non-`HandlerError` failure — IO error, context cancellation, daemon crash — evicts the daemon. The next `Call` for that kind spawns a fresh process.
- **`HandlerError` does not evict**: a clean error response from Python means the daemon is healthy; only the request failed.
- **Process groups**: subprocesses are spawned with `setpgid` so context cancellation kills the whole group, including grandchildren (e.g. torch DataLoader workers).
- **Graceful shutdown**: `Pool.Close` shuts each daemon's stdin (triggering a clean EOF exit in `run_daemon`), waits up to `ShutdownGrace` (default 10s), then `SIGKILL`s any holdouts.

## Adding a new worker

1. Add a Python script under `python/workers/` that calls `run_daemon(handler, init=...)`.
2. If the kind isn't already in `defaultScripts` ([pool.go](pool.go)), pass a custom `Scripts` map to `NewPool` — or add it to the default mapping if it's a permanent part of the pipeline.
3. From a River worker (or anywhere with a `*Pool`):
   ```go
   resp, err := pool.Call(ctx, "my_kind", &pyworker.Request{
       RequestID: jobID,
       Inputs:    map[string]any{...},
   })
   ```
   Apply per-call deadlines via `ctx`. The pool itself has no built-in timeout.

## Testing

Three levels, increasing in weight:

**1. Unit tests (no ML deps)** — [pool_test.go](pool_test.go) and [pyworker_test.go](pyworker_test.go) spawn a real Python interpreter running an echo handler. They cover spawn, daemon reuse, handler errors, crash respawn, and context cancellation. If `python3`/`python` isn't on PATH, tests `t.Skip` rather than fail.

```bash
go test ./internal/pyworker/ -v
```

**2. Standalone Python smoke test** — drive a worker script directly with a framed request to confirm its dependencies install and the model loads. Useful when bringing up a new ML worker before any Go wiring exists. See conversation notes for the one-liner.

**3. Real Go ↔ ML end-to-end** — gate a test on `CAST_E2E_ML=1` so it skips by default, point `NewPool` at the real `python/` root, and assert on output shape (embedding dimension, segment count, etc). First run is slow (model download + load); subsequent runs are fast because the daemon is reused.

## Caveats

- **POSIX-only**: `setpgid` and `SIGKILL` of `-pid` are Linux/macOS-only. Run tests in WSL, Docker, or Linux CI — not on Windows directly.
- **Per-kind serialization** is deliberate. Pipelining one daemon is possible but not worth the complexity when each call is seconds to minutes long. If a kind needs more throughput, add concurrency at the Go layer (multiple worker processes, each with its own `Pool`).
- **Model choice is fixed at init** for daemons. Per-request `options` are honored only in one-shot mode (`run`). Changing models means restarting the worker process.
