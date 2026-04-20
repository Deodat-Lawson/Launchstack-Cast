package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverpgxv5"

	"github.com/launchstack/cast/internal/config"
	"github.com/launchstack/cast/internal/db"
	"github.com/launchstack/cast/internal/jobs"
	"github.com/launchstack/cast/internal/pyworker"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

	cfg, err := config.Load()
	if err != nil {
		logger.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	if !cfg.IsDev() {
		logger = slog.New(slog.NewJSONHandler(os.Stderr, nil))
	}

	logger.Info("starting cast-worker", "config", cfg.String())

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := db.Connect(ctx, cfg.DatabaseURL)
	if err != nil {
		logger.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	// Apply River's embedded migrations. Idempotent.
	if err := jobs.Migrate(ctx, pool); err != nil {
		logger.Error("failed to apply river migrations", "error", err)
		os.Exit(1)
	}

	// Long-lived Python daemons for ML models. Spawned lazily on first call
	// per kind; one subprocess per kind keeps each model loaded in memory
	// across jobs. Per-call deadlines are applied by workers via context.
	pyPool := pyworker.NewPool(pyworker.PoolConfig{
		PythonBin:  cfg.PythonBin,
		PythonRoot: cfg.PythonRoot,
		Logger:     logger,
	})
	defer pyPool.Close()

	// One-shot runner for non-ML subprocess work (ffmpeg, scenedetect) where
	// model-load amortization doesn't apply.
	runner := pyworker.NewRunner(cfg.PythonBin, cfg.PythonRoot)

	workers := river.NewWorkers()
	jobs.Register(workers, jobs.Deps{
		Pool:   pool,
		Py:     pyPool,
		Runner: runner,
		Logger: logger,
	})

	client, err := river.NewClient[pgx.Tx](riverpgxv5.New(pool), &river.Config{
		Queues: map[string]river.QueueConfig{
			river.QueueDefault: {MaxWorkers: cfg.WorkerConcurrency},
		},
		Workers: workers,
		Logger:  logger,
	})
	if err != nil {
		logger.Error("failed to construct river client", "error", err)
		os.Exit(1)
	}

	if err := client.Start(ctx); err != nil {
		logger.Error("failed to start river client", "error", err)
		os.Exit(1)
	}

	logger.Info("worker ready", "concurrency", cfg.WorkerConcurrency)
	<-ctx.Done()
	logger.Info("shutting down")

	// Give in-flight jobs a window to finish cleanly. River's Stop blocks
	// until all workers return or the shutdown context expires, at which
	// point it hard-cancels remaining jobs and their leases lapse.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := client.Stop(shutdownCtx); err != nil {
		logger.Error("river stop error", "error", err)
	}

	logger.Info("worker stopped")
}
