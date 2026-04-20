package pyworker

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

const echoScript = `
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.run import run_daemon

def handler(req, state):
    inputs = req.get("inputs", {}) or {}
    mode = inputs.get("mode")
    if mode == "crash":
        sys.exit(7)
    if mode == "raise":
        raise RuntimeError("boom: " + str(inputs.get("msg", "")))
    if mode == "slow":
        import time
        time.sleep(60)
    return {"echo": inputs, "pid": os.getpid()}

if __name__ == "__main__":
    run_daemon(handler)
`

// setupPool writes a minimal common/run.py + workers/echo.py tree into a
// tempdir and returns a Pool pointed at it. Skips the test if no Python
// interpreter is on PATH.
func setupPool(t *testing.T) *Pool {
	t.Helper()

	pyBin := ""
	for _, candidate := range []string{"python3", "python"} {
		if p, err := exec.LookPath(candidate); err == nil {
			pyBin = p
			break
		}
	}
	if pyBin == "" {
		t.Skip("no python interpreter on PATH")
	}

	// Copy the real run.py so framing stays in lockstep with the Go side.
	runSrc, err := os.ReadFile(filepath.Join("..", "..", "python", "common", "run.py"))
	if err != nil {
		t.Fatalf("read run.py: %v", err)
	}

	root := t.TempDir()
	commonDir := filepath.Join(root, "common")
	workersDir := filepath.Join(root, "workers")
	for _, d := range []string{commonDir, workersDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	if err := os.WriteFile(filepath.Join(commonDir, "__init__.py"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(commonDir, "run.py"), runSrc, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workersDir, "echo.py"), []byte(echoScript), 0o644); err != nil {
		t.Fatal(err)
	}

	pool := NewPool(PoolConfig{
		PythonBin:     pyBin,
		PythonRoot:    root,
		Scripts:       map[string]string{"echo": "workers/echo.py"},
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		ShutdownGrace: 2 * time.Second,
	})
	t.Cleanup(func() { _ = pool.Close() })
	return pool
}

func decodeEcho(t *testing.T, resp *Response) (echoed map[string]any, pid float64) {
	t.Helper()
	var data struct {
		Echo map[string]any `json:"echo"`
		Pid  float64        `json:"pid"`
	}
	if err := json.Unmarshal(resp.Data, &data); err != nil {
		t.Fatalf("unmarshal data: %v (raw=%s)", err, string(resp.Data))
	}
	return data.Echo, data.Pid
}

func TestPool_HappyPathReusesDaemon(t *testing.T) {
	pool := setupPool(t)
	ctx := context.Background()

	resp1, err := pool.Call(ctx, "echo", &Request{RequestID: "a", Inputs: map[string]any{"hello": 1}})
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if !resp1.OK {
		t.Fatalf("first call not OK")
	}
	_, pid1 := decodeEcho(t, resp1)

	resp2, err := pool.Call(ctx, "echo", &Request{RequestID: "b", Inputs: map[string]any{"hello": 2}})
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	_, pid2 := decodeEcho(t, resp2)

	if pid1 == 0 || pid1 != pid2 {
		t.Fatalf("expected daemon reuse: pid1=%v pid2=%v", pid1, pid2)
	}
}

func TestPool_HandlerErrorDoesNotEvict(t *testing.T) {
	pool := setupPool(t)
	ctx := context.Background()

	// Prime the daemon.
	ok, err := pool.Call(ctx, "echo", &Request{RequestID: "1", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("prime: %v", err)
	}
	_, primePID := decodeEcho(t, ok)

	// Trigger a Python exception inside the daemon.
	resp, err := pool.Call(ctx, "echo", &Request{RequestID: "2", Inputs: map[string]any{"mode": "raise"}})
	var he *HandlerError
	if !errors.As(err, &he) {
		t.Fatalf("expected HandlerError, got %T: %v", err, err)
	}
	if resp == nil || resp.OK {
		t.Fatalf("expected non-OK response, got %+v", resp)
	}

	// Next call should go to the SAME daemon (same pid) since handler
	// errors don't evict.
	after, err := pool.Call(ctx, "echo", &Request{RequestID: "3", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("post-error call: %v", err)
	}
	_, afterPID := decodeEcho(t, after)
	if afterPID != primePID {
		t.Fatalf("daemon was evicted on handler error: primePID=%v afterPID=%v", primePID, afterPID)
	}
}

func TestPool_DaemonCrashTriggersRespawn(t *testing.T) {
	pool := setupPool(t)
	ctx := context.Background()

	// Prime.
	ok, err := pool.Call(ctx, "echo", &Request{RequestID: "1", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("prime: %v", err)
	}
	_, primePID := decodeEcho(t, ok)

	// Crash the daemon: sys.exit(7). Our side sees stdout EOF → read error.
	_, err = pool.Call(ctx, "echo", &Request{RequestID: "2", Inputs: map[string]any{"mode": "crash"}})
	if err == nil {
		t.Fatalf("expected error on crash call")
	}
	var he *HandlerError
	if errors.As(err, &he) {
		t.Fatalf("crash should not surface as HandlerError: %v", err)
	}

	// Next call respawns. Different PID proves it.
	after, err := pool.Call(ctx, "echo", &Request{RequestID: "3", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("post-crash call: %v", err)
	}
	_, afterPID := decodeEcho(t, after)
	if afterPID == 0 || afterPID == primePID {
		t.Fatalf("expected fresh daemon: primePID=%v afterPID=%v", primePID, afterPID)
	}
}

func TestPool_ContextCancelKillsDaemon(t *testing.T) {
	pool := setupPool(t)

	// Prime.
	bg := context.Background()
	ok, err := pool.Call(bg, "echo", &Request{RequestID: "1", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("prime: %v", err)
	}
	_, primePID := decodeEcho(t, ok)

	// Slow handler + short deadline.
	ctx, cancel := context.WithTimeout(bg, 200*time.Millisecond)
	defer cancel()
	_, err = pool.Call(ctx, "echo", &Request{RequestID: "2", Inputs: map[string]any{"mode": "slow"}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}

	// Daemon should be evicted. Next call spawns a new one.
	after, err := pool.Call(bg, "echo", &Request{RequestID: "3", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("post-timeout call: %v", err)
	}
	_, afterPID := decodeEcho(t, after)
	if afterPID == primePID {
		t.Fatalf("daemon was not respawned after timeout: primePID=%v afterPID=%v", primePID, afterPID)
	}
}
