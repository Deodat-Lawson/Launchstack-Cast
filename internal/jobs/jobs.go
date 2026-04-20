// Package jobs defines River job types and workers for every stage of the
// Cast pipeline. Job args are serialized to JSON and survive process restarts;
// workers dispatch into the per-stage packages under internal/.
package jobs

import (
	"context"
	"errors"
	"log/slog"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverpgxv5"
	"github.com/riverqueue/river/rivermigrate"

	"github.com/launchstack/cast/internal/pyworker"
)

// Deps are the shared dependencies each worker needs. A single Deps is built
// at process startup and handed to every worker constructor.
type Deps struct {
	Pool   *pgxpool.Pool
	Py     *pyworker.Pool   // long-lived ML daemons (whisper, clip, faces)
	Runner *pyworker.Runner // one-shot subprocess (ffmpeg, scenedetect)
	Logger *slog.Logger
}

// Register adds every worker in this package to the given river.Workers.
// Call this once when building the river.Client.
func Register(workers *river.Workers, deps Deps) {
	river.AddWorker(workers, &IngestWorker{deps: deps})
	river.AddWorker(workers, &TranscribeWorker{deps: deps})
	river.AddWorker(workers, &VisualEmbedWorker{deps: deps})
	river.AddWorker(workers, &FacesWorker{deps: deps})
}

// Migrate applies River's embedded schema migrations. Idempotent — safe to
// call on every process start.
func Migrate(ctx context.Context, pool *pgxpool.Pool) error {
	migrator, err := rivermigrate.New(riverpgxv5.New(pool), nil)
	if err != nil {
		return err
	}
	_, err = migrator.Migrate(ctx, rivermigrate.DirectionUp, nil)
	return err
}

// ErrNotImplemented is returned by stage workers whose underlying package is
// still a stub. River treats this as a retryable failure; with MaxAttempts
// capped below, a job stuck on a stub will end up in river's failed state
// (discarded) rather than retrying forever.
var ErrNotImplemented = errors.New("stage not implemented")

// -----------------------------------------------------------------------------
// Ingest: ffmpeg audio extract + PySceneDetect scene split + keyframe extract.
// Inserts one clips row per detected scene, then enqueues perception jobs.
// -----------------------------------------------------------------------------

type IngestArgs struct {
	VideoID uuid.UUID `json:"video_id"`
}

func (IngestArgs) Kind() string { return "ingest" }

func (IngestArgs) InsertOpts() river.InsertOpts {
	return river.InsertOpts{
		MaxAttempts: 3,
		UniqueOpts: river.UniqueOpts{
			ByArgs: true,
		},
	}
}

type IngestWorker struct {
	river.WorkerDefaults[IngestArgs]
	deps Deps
}

func (w *IngestWorker) Work(ctx context.Context, job *river.Job[IngestArgs]) error {
	w.deps.Logger.Info("ingest", "video_id", job.Args.VideoID, "attempt", job.Attempt)
	return ErrNotImplemented
}

// -----------------------------------------------------------------------------
// Transcribe: whisper over a clip's audio range.
// -----------------------------------------------------------------------------

type TranscribeArgs struct {
	ClipID    uuid.UUID `json:"clip_id"`
	AudioPath string    `json:"audio_path"`
	Language  string    `json:"language,omitempty"`
}

func (TranscribeArgs) Kind() string { return "transcribe" }

func (TranscribeArgs) InsertOpts() river.InsertOpts {
	return river.InsertOpts{MaxAttempts: 3}
}

type TranscribeWorker struct {
	river.WorkerDefaults[TranscribeArgs]
	deps Deps
}

func (w *TranscribeWorker) Work(ctx context.Context, job *river.Job[TranscribeArgs]) error {
	w.deps.Logger.Info("transcribe", "clip_id", job.Args.ClipID, "attempt", job.Attempt)
	return ErrNotImplemented
}

// -----------------------------------------------------------------------------
// VisualEmbed: OpenCLIP embedding over a clip's keyframe.
// -----------------------------------------------------------------------------

type VisualEmbedArgs struct {
	ClipID       uuid.UUID `json:"clip_id"`
	KeyframePath string    `json:"keyframe_path"`
}

func (VisualEmbedArgs) Kind() string { return "visual_embed" }

func (VisualEmbedArgs) InsertOpts() river.InsertOpts {
	return river.InsertOpts{MaxAttempts: 3}
}

type VisualEmbedWorker struct {
	river.WorkerDefaults[VisualEmbedArgs]
	deps Deps
}

func (w *VisualEmbedWorker) Work(ctx context.Context, job *river.Job[VisualEmbedArgs]) error {
	w.deps.Logger.Info("visual_embed", "clip_id", job.Args.ClipID, "attempt", job.Attempt)
	return ErrNotImplemented
}

// -----------------------------------------------------------------------------
// Faces: InsightFace detection + ArcFace embedding over a clip's keyframe.
// -----------------------------------------------------------------------------

type FacesArgs struct {
	ClipID       uuid.UUID `json:"clip_id"`
	KeyframePath string    `json:"keyframe_path"`
}

func (FacesArgs) Kind() string { return "faces" }

func (FacesArgs) InsertOpts() river.InsertOpts {
	return river.InsertOpts{MaxAttempts: 3}
}

type FacesWorker struct {
	river.WorkerDefaults[FacesArgs]
	deps Deps
}

func (w *FacesWorker) Work(ctx context.Context, job *river.Job[FacesArgs]) error {
	w.deps.Logger.Info("faces", "clip_id", job.Args.ClipID, "attempt", job.Attempt)
	return ErrNotImplemented
}
