package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
)

type Config struct {
	// Server
	Addr string `env:"CAST_ADDR" envDefault:":8080"`
	Env  string `env:"CAST_ENV" envDefault:"dev"`

	// Database
	DatabaseURL string `env:"DATABASE_URL,required"`

	// Data directory for video files and keyframes
	DataDir string `env:"CAST_DATA_DIR" envDefault:"./data"`

	// Web directory containing the frontend prototype (Cast.html + JSX).
	// Served at / by the API.
	WebDir string `env:"CAST_WEB_DIR" envDefault:"./web"`

	// Python
	PythonBin  string `env:"CAST_PYTHON_BIN" envDefault:"python3"`
	PythonRoot string `env:"CAST_PYTHON_ROOT" envDefault:"./python"`

	// LLM
	AnthropicAPIKey string `env:"ANTHROPIC_API_KEY"`

	// Worker: max in-flight jobs this process will run. River handles poll
	// cadence internally via LISTEN/NOTIFY + a sane fallback, so there's no
	// poll-interval knob here.
	WorkerConcurrency int `env:"CAST_WORKER_CONCURRENCY" envDefault:"2"`

	// Python worker timeouts
	TimeoutTranscribe  time.Duration `env:"CAST_PY_TIMEOUT_TRANSCRIBE" envDefault:"300s"`
	TimeoutVisualEmbed time.Duration `env:"CAST_PY_TIMEOUT_VISUAL_EMBED" envDefault:"120s"`
	TimeoutFaces       time.Duration `env:"CAST_PY_TIMEOUT_FACES" envDefault:"120s"`
}

func Load() (*Config, error) {
	cfg := &Config{}
	if err := env.Parse(cfg); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	return cfg, nil
}

func (c *Config) IsDev() bool {
	return c.Env == "dev"
}

// String returns a redacted representation safe for logging.
func (c *Config) String() string {
	redact := func(s string) string {
		if s == "" {
			return "(unset)"
		}
		return "***"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "addr=%s env=%s ", c.Addr, c.Env)
	fmt.Fprintf(&b, "database_url=%s ", redact(c.DatabaseURL))
	fmt.Fprintf(&b, "data_dir=%s ", c.DataDir)
	fmt.Fprintf(&b, "python_bin=%s python_root=%s ", c.PythonBin, c.PythonRoot)
	fmt.Fprintf(&b, "anthropic_api_key=%s ", redact(c.AnthropicAPIKey))
	fmt.Fprintf(&b, "worker_concurrency=%d", c.WorkerConcurrency)
	return b.String()
}
