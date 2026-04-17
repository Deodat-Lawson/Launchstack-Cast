"""
Shared harness for Cast Python workers.

Every worker script calls `run(handler_fn)` which:
  1. Reads a JSON request from stdin.
  2. Calls handler_fn(request) -> data dict.
  3. Writes a JSON response to stdout.
  4. On exception, writes a structured error to stdout and exits 1.

Logs go to stderr so they don't pollute the JSON protocol.
"""

import json
import sys
import time
import traceback


def run(handler):
    """
    Entry point for a worker script.

    handler: callable(request: dict) -> dict
        Receives the full request envelope. Must return a dict that becomes
        response["data"].
    """
    start = time.monotonic()
    request = None

    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        request_id = request.get("request_id", "")

        data = handler(request)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        response = {
            "request_id": request_id,
            "ok": True,
            "data": data,
            "error": None,
            "meta": {"duration_ms": elapsed_ms},
        }
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        request_id = request.get("request_id", "") if request else ""

        # Traceback to stderr for debugging.
        traceback.print_exc(file=sys.stderr)

        error_code = type(exc).__name__
        response = {
            "request_id": request_id,
            "ok": False,
            "data": None,
            "error": {"code": error_code, "message": str(exc)},
            "meta": {"duration_ms": elapsed_ms},
        }
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()
        sys.exit(1)
