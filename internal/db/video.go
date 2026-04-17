package db

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Video struct {
	ID          uuid.UUID  `json:"id"`
	SourceURL   *string    `json:"source_url"`
	SourcePath  string     `json:"source_path"`
	DurationSec *float64   `json:"duration_sec"`
	Status      string     `json:"status"`
	Error       *string    `json:"error"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

func InsertVideo(ctx context.Context, pool *pgxpool.Pool, sourceURL *string, sourcePath string) (*Video, error) {
	v := &Video{}
	err := pool.QueryRow(ctx,
		`INSERT INTO videos (source_url, source_path)
		 VALUES ($1, $2)
		 RETURNING id, source_url, source_path, duration_sec, status, error, created_at, updated_at`,
		sourceURL, sourcePath,
	).Scan(&v.ID, &v.SourceURL, &v.SourcePath, &v.DurationSec, &v.Status, &v.Error, &v.CreatedAt, &v.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("db: insert video: %w", err)
	}
	return v, nil
}

func GetVideo(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (*Video, error) {
	v := &Video{}
	err := pool.QueryRow(ctx,
		`SELECT id, source_url, source_path, duration_sec, status, error, created_at, updated_at
		 FROM videos WHERE id = $1`,
		id,
	).Scan(&v.ID, &v.SourceURL, &v.SourcePath, &v.DurationSec, &v.Status, &v.Error, &v.CreatedAt, &v.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("db: get video: %w", err)
	}
	return v, nil
}
