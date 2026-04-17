package pyworker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// Request is the envelope sent to a Python worker on stdin.
type Request struct {
	RequestID string      `json:"request_id"`
	Inputs    interface{} `json:"inputs"`
	Options   interface{} `json:"options,omitempty"`
}

// Response is the envelope read from a Python worker on stdout.
type Response struct {
	RequestID string           `json:"request_id"`
	OK        bool             `json:"ok"`
	Data      json.RawMessage  `json:"data"`
	Error     *ResponseError   `json:"error"`
	Meta      *ResponseMeta    `json:"meta"`
}

type ResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type ResponseMeta struct {
	DurationMs int    `json:"duration_ms"`
	Model      string `json:"model,omitempty"`
}

// Runner executes Python worker scripts as subprocesses.
type Runner struct {
	PythonBin  string
	PythonRoot string
}

func NewRunner(pythonBin, pythonRoot string) *Runner {
	return &Runner{
		PythonBin:  pythonBin,
		PythonRoot: pythonRoot,
	}
}

// Run invokes a Python worker script with the given request and timeout.
// The script path is relative to PythonRoot (e.g. "workers/transcribe.py").
func (r *Runner) Run(ctx context.Context, script string, timeout time.Duration, req *Request) (*Response, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	scriptPath := filepath.Join(r.PythonRoot, script)

	inputBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("pyworker: marshal request: %w", err)
	}

	cmd := exec.CommandContext(ctx, r.PythonBin, "-u", scriptPath)
	cmd.Stdin = bytes.NewReader(inputBytes)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Kill the entire process group on timeout so child processes don't linger.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	}
	cmd.WaitDelay = 5 * time.Second

	if err := cmd.Run(); err != nil {
		stderrTail := stderr.String()
		if len(stderrTail) > 2000 {
			stderrTail = stderrTail[len(stderrTail)-2000:]
		}
		return nil, fmt.Errorf("pyworker: %s failed (exit=%v): %s", script, err, stderrTail)
	}

	var resp Response
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("pyworker: unmarshal response from %s: %w (stdout=%q)", script, err, stdout.String())
	}

	if !resp.OK {
		msg := "unknown error"
		if resp.Error != nil {
			msg = resp.Error.Message
		}
		return &resp, fmt.Errorf("pyworker: %s returned error: %s", script, msg)
	}

	return &resp, nil
}
