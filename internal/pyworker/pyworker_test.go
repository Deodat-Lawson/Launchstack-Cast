package pyworker

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testdataRoot(t *testing.T) string {
	t.Helper()
	// Walk up from internal/pyworker/ to repo root.
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(wd, "..", "..", "testdata")
}

func TestRunEcho(t *testing.T) {
	root := testdataRoot(t)
	runner := NewRunner("python3", root)

	req := &Request{
		RequestID: "test-123",
		Inputs: map[string]string{
			"greeting": "hello",
		},
	}

	resp, err := runner.Run(context.Background(), "workers/echo.py", 10*time.Second, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.RequestID != "test-123" {
		t.Errorf("request_id = %q, want %q", resp.RequestID, "test-123")
	}
	if !resp.OK {
		t.Errorf("ok = false, want true")
	}

	var data map[string]string
	if err := json.Unmarshal(resp.Data, &data); err != nil {
		t.Fatalf("unmarshal data: %v", err)
	}
	if data["greeting"] != "hello" {
		t.Errorf("data[greeting] = %q, want %q", data["greeting"], "hello")
	}
}

func TestRunFailure(t *testing.T) {
	root := testdataRoot(t)
	runner := NewRunner("python3", root)

	req := &Request{
		RequestID: "test-fail",
		Inputs:    map[string]string{},
	}

	resp, err := runner.Run(context.Background(), "workers/fail.py", 10*time.Second, req)
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// The response should still be populated with error details.
	if resp == nil {
		t.Fatal("expected non-nil response on failure")
	}
	if resp.OK {
		t.Error("ok = true, want false")
	}
	if resp.Error == nil {
		t.Fatal("error field is nil")
	}
	if resp.Error.Code != "ValueError" {
		t.Errorf("error.code = %q, want %q", resp.Error.Code, "ValueError")
	}
}

func TestRunTimeout(t *testing.T) {
	root := testdataRoot(t)
	runner := NewRunner("python3", root)

	// Use a very short timeout with the echo script — it should complete,
	// but a sleep-based script would fail. We test the mechanism works by
	// using a context that's already cancelled.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := &Request{RequestID: "test-timeout", Inputs: map[string]string{}}
	_, err := runner.Run(ctx, "workers/echo.py", 1*time.Millisecond, req)
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}
