package api

import (
	"log/slog"
	"net/http"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"
)

// NewServer wires up routes and middleware, returning a ready-to-serve handler.
func NewServer(pool *pgxpool.Pool, riverClient *river.Client[pgx.Tx], logger *slog.Logger, dataDir, webDir string) http.Handler {
	h := NewHandlers(pool, riverClient, logger, dataDir)

	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/videos", h.CreateVideo)
	mux.HandleFunc("GET /v1/videos/{id}", h.GetVideo)
	mux.HandleFunc("GET /healthz", h.Healthz)
	mux.HandleFunc("GET /readyz", h.Readyz)

	// Frontend prototype. Root redirects to Cast.html (the design ships
	// under that name); everything else falls through to the file server.
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/Cast.html", http.StatusFound)
	})
	mux.Handle("GET /", http.FileServer(http.Dir(webDir)))

	// Middleware chain: outermost runs first.
	var handler http.Handler = mux
	handler = loggingMiddleware(logger)(handler)
	handler = requestIDMiddleware(handler)

	return handler
}
