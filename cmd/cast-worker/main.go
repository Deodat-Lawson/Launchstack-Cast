package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/launchstack/cast/internal/config"
	"github.com/launchstack/cast/internal/db"
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

	// TODO: implement job polling loop
	logger.Info("worker ready", "concurrency", cfg.WorkerConcurrency, "poll_interval", cfg.WorkerPollInterval)
	<-ctx.Done()
	logger.Info("worker stopped")
}
