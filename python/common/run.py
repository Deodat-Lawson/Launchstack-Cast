"""
Shared harness for Cast Python workers.

Two entry points:

  run(handler)        — one-shot. Reads a single JSON request from stdin,
                        writes one response, exits. Good for CLI testing and
                        anything that doesn't warrant a persistent process
                        (e.g. ffmpeg shell-outs). This is the original
                        contract; unchanged.

  run_daemon(handler, init=None)
                      — long-lived. Optionally runs init() once to load
                        models, then loops: read one framed JSON request,
                        call handler, write one framed JSON response. A
                        handler exception does NOT kill the daemon; an error
                        response is sent and the loop continues.

Wire format for run_daemon:
  4-byte big-endian length + UTF-8 JSON body, on stdin and stdout.
  Length-prefix framing sidesteps stdout-buffering ambiguity and makes
  message boundaries unambiguous even if handlers accidentally print.

Logs go to stderr so they don't pollute the JSON protocol.
"""

import json
import os
import struct
import sys
import time
import traceback


def run(handler):
    """
    One-shot entry point. Reads a single JSON request from stdin, calls
    handler(request) -> dict, writes the JSON response to stdout, and exits.
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


_LEN_STRUCT = struct.Struct(">I")  # 4-byte big-endian unsigned


def _read_frame(fd):
    """Read one length-prefixed JSON frame from a binary fd. Returns the
    decoded dict, or None on clean EOF (parent closed stdin)."""
    header = _readn(fd, _LEN_STRUCT.size)
    if header is None:
        return None
    (length,) = _LEN_STRUCT.unpack(header)
    body = _readn(fd, length)
    if body is None:
        raise EOFError("truncated frame: expected %d bytes of body" % length)
    return json.loads(body.decode("utf-8"))


def _readn(fd, n):
    """Read exactly n bytes from fd, or return None if EOF hit at frame start."""
    buf = bytearray()
    while len(buf) < n:
        chunk = fd.read(n - len(buf))
        if not chunk:
            return None if not buf else b""
        buf.extend(chunk)
    return bytes(buf)


def _write_frame(fd, payload):
    body = json.dumps(payload).encode("utf-8")
    fd.write(_LEN_STRUCT.pack(len(body)))
    fd.write(body)
    fd.flush()


def run_daemon(handler, init=None):
    """
    Long-lived entry point. Runs init() once (if provided), then loops over
    framed requests on stdin, writing framed responses to stdout.

    handler: callable(request: dict, state: any) -> dict
        Receives the request envelope and whatever init() returned (or None).
    init: callable() -> any (optional)
        Called once at startup. Use to pre-load models. Its return value is
        passed to every handler call as the second argument.
    """
    # Binary stdio. We bypass any text-mode buffering to make framing safe.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    state = None
    if init is not None:
        t0 = time.monotonic()
        state = init()
        sys.stderr.write(
            "pyworker: init complete in %d ms (pid=%d)\n"
            % (int((time.monotonic() - t0) * 1000), os.getpid())
        )
        sys.stderr.flush()

    while True:
        try:
            request = _read_frame(stdin)
        except EOFError as exc:
            sys.stderr.write("pyworker: framing error: %s\n" % exc)
            sys.stderr.flush()
            sys.exit(2)

        if request is None:
            # Clean EOF — parent closed stdin. Exit 0.
            return

        start = time.monotonic()
        request_id = request.get("request_id", "") if isinstance(request, dict) else ""

        try:
            data = handler(request, state)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            _write_frame(
                stdout,
                {
                    "request_id": request_id,
                    "ok": True,
                    "data": data,
                    "error": None,
                    "meta": {"duration_ms": elapsed_ms},
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            traceback.print_exc(file=sys.stderr)
            _write_frame(
                stdout,
                {
                    "request_id": request_id,
                    "ok": False,
                    "data": None,
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                    "meta": {"duration_ms": elapsed_ms},
                },
            )
            # Do not exit — the daemon stays up for the next request.
