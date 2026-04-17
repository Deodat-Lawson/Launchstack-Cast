package api

import (
	"log/slog"
	"net/http"

	"github.com/jackc/pgx/v5/pgxpool"
)

// NewServer wires up routes and middleware, returning a ready-to-serve handler.
func NewServer(pool *pgxpool.Pool, logger *slog.Logger, dataDir string) http.Handler {
	h := NewHandlers(pool, logger, dataDir)

	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/videos", h.CreateVideo)
	mux.HandleFunc("GET /v1/videos/{id}", h.GetVideo)
	mux.HandleFunc("GET /healthz", h.Healthz)
	mux.HandleFunc("GET /readyz", h.Readyz)

	// Middleware chain: outermost runs first.
	var handler http.Handler = mux
	handler = loggingMiddleware(logger)(handler)
	handler = requestIDMiddleware(handler)

	return handler
}
