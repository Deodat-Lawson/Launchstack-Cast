-- +goose Up

-- River supersedes our hand-rolled queue table. River's schema is applied
-- programmatically at app startup via rivermigrate, so we only need to drop
-- the legacy table here.
DROP TABLE IF EXISTS jobs;

-- +goose Down

CREATE TABLE jobs (
    id         bigserial PRIMARY KEY,
    video_id   uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    kind       text NOT NULL,
    payload    jsonb NOT NULL DEFAULT '{}'::jsonb,
    status     text NOT NULL DEFAULT 'pending',
    attempts   int NOT NULL DEFAULT 0,
    last_error text,
    locked_by  text,
    locked_at  timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX jobs_pending_idx ON jobs(status, kind) WHERE status = 'pending';
