package pyworker

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// Pool manages long-lived Python "daemon" subprocesses, one per kind. Each
// daemon loads its ML model once at startup and then serves framed JSON
// requests from stdin in a loop (see python/common/run.py:run_daemon).
//
// Goals:
//   - Amortize model load time (Whisper/CLIP/InsightFace take 10–30s to load).
//   - Crash-tolerant: a dead daemon is replaced on the next Call to that kind.
//   - ctx-honoring: a timed-out Call kills the subprocess group so pipes
//     don't back up with a stale response; the next Call respawns.
//   - Serial per daemon: one in-flight Call per kind. Multiple kinds run in
//     parallel. (Pipelining one daemon is possible but not worth the
//     complexity when each Call is seconds to minutes long.)
type Pool struct {
	cfg PoolConfig

	mu      sync.Mutex
	daemons map[string]*daemon
	closed  bool
}

type PoolConfig struct {
	PythonBin  string
	PythonRoot string
	Logger     *slog.Logger

	// Scripts overrides the default kind → script mapping. If nil, the
	// built-in mapping is used (transcribe, visual_embed, faces).
	Scripts map[string]string

	// ShutdownGrace is how long Close waits for a daemon to exit after
	// closing its stdin before SIGKILL-ing it. Default 10s.
	ShutdownGrace time.Duration
}

var defaultScripts = map[string]string{
	"transcribe":   "workers/transcribe.py",
	"visual_embed": "workers/visual_embed.py",
	"faces":        "workers/faces.py",
}

func NewPool(cfg PoolConfig) *Pool {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.ShutdownGrace == 0 {
		cfg.ShutdownGrace = 10 * time.Second
	}
	if cfg.Scripts == nil {
		cfg.Scripts = defaultScripts
	}
	return &Pool{cfg: cfg, daemons: make(map[string]*daemon)}
}

// Call serializes one request through the daemon for the given kind,
// spawning the daemon on first use. A failed Call (timeout, IO error, non-OK
// response from a dead daemon) evicts the daemon so the next Call respawns.
//
// Concurrency: Calls for the same kind serialize on the daemon's mutex.
// Calls for different kinds run in parallel.
func (p *Pool) Call(ctx context.Context, kind string, req *Request) (*Response, error) {
	d, err := p.getOrSpawn(kind)
	if err != nil {
		return nil, err
	}

	resp, err := d.call(ctx, req)
	if err != nil {
		// A HandlerError means Python raised an exception but the daemon
		// stayed up — don't evict. Anything else (IO error, ctx timeout,
		// crash) leaves the daemon in an unknown state, so evict and let
		// the next Call respawn.
		var he *HandlerError
		if !errors.As(err, &he) {
			p.evict(kind, d)
		}
	}
	return resp, err
}

// Close asks every daemon to exit cleanly, then kills any that don't. Safe
// to call multiple times.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	daemons := make([]*daemon, 0, len(p.daemons))
	for _, d := range p.daemons {
		daemons = append(daemons, d)
	}
	p.daemons = nil
	p.mu.Unlock()

	var wg sync.WaitGroup
	for _, d := range daemons {
		wg.Add(1)
		go func(d *daemon) {
			defer wg.Done()
			d.shutdown(p.cfg.ShutdownGrace)
		}(d)
	}
	wg.Wait()
	return nil
}

func (p *Pool) getOrSpawn(kind string) (*daemon, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, errors.New("pyworker: pool is closed")
	}
	if d, ok := p.daemons[kind]; ok {
		return d, nil
	}

	script, ok := p.cfg.Scripts[kind]
	if !ok {
		return nil, fmt.Errorf("pyworker: unknown kind %q", kind)
	}

	d, err := spawnDaemon(p.cfg.PythonBin, filepath.Join(p.cfg.PythonRoot, script), kind, p.cfg.Logger)
	if err != nil {
		return nil, err
	}
	p.daemons[kind] = d
	return d, nil
}

func (p *Pool) evict(kind string, d *daemon) {
	p.mu.Lock()
	if cur, ok := p.daemons[kind]; ok && cur == d {
		delete(p.daemons, kind)
	}
	p.mu.Unlock()
	// Best-effort cleanup; we already know something went wrong. Do this
	// outside the pool lock because shutdown can take up to ShutdownGrace.
	go d.shutdown(p.cfg.ShutdownGrace)
}

// -----------------------------------------------------------------------------
// daemon: one persistent subprocess.
// -----------------------------------------------------------------------------

type daemon struct {
	kind   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	logger *slog.Logger

	mu sync.Mutex // serializes calls through this daemon
}

func spawnDaemon(pythonBin, scriptPath, kind string, logger *slog.Logger) (*daemon, error) {
	cmd := exec.Command(pythonBin, "-u", scriptPath)

	// Kill the entire process group on shutdown so grandchildren (e.g.
	// torch DataLoader workers) don't survive. Mirrors Runner.Run.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("pyworker: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("pyworker: stdout pipe: %w", err)
	}
	// Inherit stderr so Python tracebacks and the init banner go straight
	// to the worker log. (Default nil would send them to /dev/null.)
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("pyworker: start %s: %w", scriptPath, err)
	}

	logger.Info("pyworker daemon spawned", "kind", kind, "pid", cmd.Process.Pid, "script", scriptPath)

	return &daemon{
		kind:   kind,
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
		logger: logger,
	}, nil
}

type readResult struct {
	resp *Response
	err  error
}

func (d *daemon) call(ctx context.Context, req *Request) (*Response, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if err := writeFrame(d.stdin, req); err != nil {
		return nil, fmt.Errorf("pyworker: write frame (%s): %w", d.kind, err)
	}

	ch := make(chan readResult, 1)
	go func() {
		resp, err := readFrame(d.stdout)
		ch <- readResult{resp, err}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return nil, fmt.Errorf("pyworker: read frame (%s): %w", d.kind, r.err)
		}
		if !r.resp.OK {
			msg := "unknown error"
			if r.resp.Error != nil {
				msg = r.resp.Error.Message
			}
			// A Python-side handler exception. The daemon itself is
			// still healthy — don't trigger eviction. Signal this by
			// returning the response AND the error (callers can inspect
			// either); the Pool.Call eviction check is on err != nil,
			// which is a design tradeoff. We want eviction on IO / crash,
			// NOT on a clean error response. Wrap in a sentinel so Pool
			// can distinguish.
			return r.resp, &HandlerError{Kind: d.kind, Message: msg}
		}
		return r.resp, nil
	case <-ctx.Done():
		// Kill the subprocess group. The read goroutine will unblock
		// when stdout closes; we drain it before returning to avoid
		// leaking a goroutine (and more importantly, leaking access to
		// a pipe we've stopped reading).
		d.kill()
		<-ch
		return nil, ctx.Err()
	}
}

// shutdown closes stdin (triggering a clean EOF exit in run_daemon) and
// waits up to grace. If the process hasn't exited by then, kill the group.
func (d *daemon) shutdown(grace time.Duration) {
	_ = d.stdin.Close()

	done := make(chan error, 1)
	go func() { done <- d.cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			d.logger.Info("pyworker daemon exited", "kind", d.kind, "error", err)
		}
	case <-time.After(grace):
		d.logger.Warn("pyworker daemon did not exit in grace, killing", "kind", d.kind, "pid", d.cmd.Process.Pid)
		d.kill()
		<-done
	}
}

func (d *daemon) kill() {
	if d.cmd.Process == nil {
		return
	}
	// Negative PID = process group.
	_ = syscall.Kill(-d.cmd.Process.Pid, syscall.SIGKILL)
}

// HandlerError means the daemon returned a structured error response. The
// daemon itself is healthy and should not be evicted.
type HandlerError struct {
	Kind    string
	Message string
}

func (e *HandlerError) Error() string {
	return fmt.Sprintf("pyworker: %s handler error: %s", e.Kind, e.Message)
}

// -----------------------------------------------------------------------------
// Framing: 4-byte BE length prefix + JSON body. Matches python/common/run.py.
// -----------------------------------------------------------------------------

func writeFrame(w io.Writer, v any) error {
	body, err := json.Marshal(v)
	if err != nil {
		return err
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(body)))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	_, err = w.Write(body)
	return err
}

func readFrame(r io.Reader) (*Response, error) {
	var header [4]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, err
	}
	length := binary.BigEndian.Uint32(header[:])
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, fmt.Errorf("short body: %w", err)
	}
	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &resp, nil
}
