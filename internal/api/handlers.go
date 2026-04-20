package api

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"github.com/launchstack/cast/internal/db"
	"github.com/launchstack/cast/internal/jobs"
)

type Handlers struct {
	pool    *pgxpool.Pool
	river   *river.Client[pgx.Tx]
	logger  *slog.Logger
	dataDir string
}

func NewHandlers(pool *pgxpool.Pool, riverClient *river.Client[pgx.Tx], logger *slog.Logger, dataDir string) *Handlers {
	return &Handlers{pool: pool, river: riverClient, logger: logger, dataDir: dataDir}
}

// POST /v1/videos
func (h *Handlers) CreateVideo(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SourceURL string `json:"source_url"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", "could not parse request body")
		return
	}

	if req.SourceURL == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "source_url is required")
		return
	}

	// Insert video + enqueue ingest job atomically: either both land or
	// neither does. A committed videos row without a queued job would be a
	// ghost that the worker never picks up.
	tx, err := h.pool.Begin(r.Context())
	if err != nil {
		h.logger.Error("failed to begin tx", "error", err, "request_id", RequestID(r.Context()))
		writeError(w, http.StatusInternalServerError, "db_error", "failed to create video")
		return
	}
	defer tx.Rollback(r.Context()) //nolint:errcheck // safe to ignore after commit

	// For v1, source_path is derived from the video ID after creation.
	// Insert with a placeholder, then set the real path below.
	video, err := db.InsertVideo(r.Context(), tx, &req.SourceURL, "pending")
	if err != nil {
		h.logger.Error("failed to insert video", "error", err, "request_id", RequestID(r.Context()))
		writeError(w, http.StatusInternalServerError, "db_error", "failed to create video")
		return
	}

	if _, err := h.river.InsertTx(r.Context(), tx, jobs.IngestArgs{VideoID: video.ID}, nil); err != nil {
		h.logger.Error("failed to enqueue ingest job", "error", err, "video_id", video.ID)
		writeError(w, http.StatusInternalServerError, "queue_error", "failed to enqueue ingest job")
		return
	}

	if err := tx.Commit(r.Context()); err != nil {
		h.logger.Error("failed to commit", "error", err, "video_id", video.ID)
		writeError(w, http.StatusInternalServerError, "db_error", "failed to create video")
		return
	}

	// Side effects after commit: data dir is keyed by video ID. If mkdir
	// fails here, the row + job still exist; the ingest worker will fail
	// loudly, which is the right signal (disk problem).
	sourcePath := filepath.Join(h.dataDir, video.ID.String())
	if err := os.MkdirAll(sourcePath, 0o755); err != nil {
		h.logger.Error("failed to create data dir", "error", err, "video_id", video.ID)
		writeError(w, http.StatusInternalServerError, "fs_error", "failed to create data directory")
		return
	}

	video.SourcePath = sourcePath

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(video)
}

// GET /v1/videos/{id}
func (h *Handlers) GetVideo(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_id", "video id must be a valid UUID")
		return
	}

	video, err := db.GetVideo(r.Context(), h.pool, id)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusNotFound, "not_found", "video not found")
			return
		}
		h.logger.Error("failed to get video", "error", err, "request_id", RequestID(r.Context()))
		writeError(w, http.StatusInternalServerError, "db_error", "failed to fetch video")
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(video)
}

// GET /healthz
func (h *Handlers) Healthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// GET /readyz
func (h *Handlers) Readyz(w http.ResponseWriter, r *http.Request) {
	if err := h.pool.Ping(r.Context()); err != nil {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "unavailable", "error": err.Error()})
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
}

type apiError struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	e := apiError{}
	e.Error.Code = code
	e.Error.Message = message
	json.NewEncoder(w).Encode(e)
}
