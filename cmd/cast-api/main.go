package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverpgxv5"

	"github.com/launchstack/cast/internal/api"
	"github.com/launchstack/cast/internal/config"
	"github.com/launchstack/cast/internal/db"
	"github.com/launchstack/cast/internal/jobs"
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

	logger.Info("starting cast-api", "config", cfg.String())

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := db.Connect(ctx, cfg.DatabaseURL)
	if err != nil {
		logger.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	// Apply River's embedded migrations before constructing the client. Safe
	// to run on every start; cast-worker does the same.
	if err := jobs.Migrate(ctx, pool); err != nil {
		logger.Error("failed to apply river migrations", "error", err)
		os.Exit(1)
	}

	// Insert-only River client: no Queues/Workers configured, never Started.
	// Used only to InsertTx jobs alongside video creation.
	riverClient, err := river.NewClient[pgx.Tx](riverpgxv5.New(pool), &river.Config{})
	if err != nil {
		logger.Error("failed to construct river client", "error", err)
		os.Exit(1)
	}

	handler := api.NewServer(pool, riverClient, logger, cfg.DataDir, cfg.WebDir)

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in a goroutine so we can listen for shutdown signals.
	go func() {
		logger.Info("listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	logger.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("shutdown error", "error", err)
		os.Exit(1)
	}

	logger.Info("stopped")
}
