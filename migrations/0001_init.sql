-- +goose Up

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- A submitted video. Row created the moment a job is accepted.
CREATE TABLE videos (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url  text,
    source_path text NOT NULL,
    duration_sec double precision,
    status      text NOT NULL DEFAULT 'pending',
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Detected scenes. One row per PySceneDetect cut.
CREATE TABLE clips (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id         uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx              int NOT NULL,
    start_sec        double precision NOT NULL,
    end_sec          double precision NOT NULL,
    keyframe_path    text,
    caption          text,
    transcript       text,
    transcript_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(transcript, ''))) STORED,
    visual_embed     vector(512),
    transcript_embed vector(1024),
    UNIQUE (video_id, idx)
);

CREATE INDEX clips_visual_embed_idx ON clips USING ivfflat (visual_embed vector_cosine_ops) WITH (lists = 100);
CREATE INDEX clips_transcript_tsv_idx ON clips USING gin (transcript_tsv);

-- Per-clip whisper segments with timestamps.
CREATE TABLE transcript_segments (
    id        bigserial PRIMARY KEY,
    clip_id   uuid NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    start_sec double precision NOT NULL,
    end_sec   double precision NOT NULL,
    text      text NOT NULL
);

CREATE INDEX transcript_segments_clip_time_idx ON transcript_segments(clip_id, start_sec);

-- Stable identity across a video.
CREATE TABLE entities (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id       uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    label          text,
    centroid_embed vector(512),
    n_detections   int NOT NULL DEFAULT 0,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX entities_video_idx ON entities(video_id);
CREATE INDEX entities_label_trgm_idx ON entities USING gin (label gin_trgm_ops);

-- Raw face detections from keyframes. entity_id filled after clustering.
CREATE TABLE face_detections (
    id        bigserial PRIMARY KEY,
    clip_id   uuid NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    frame_sec double precision NOT NULL,
    bbox      int[] NOT NULL,
    embedding vector(512) NOT NULL,
    det_score real NOT NULL,
    entity_id uuid REFERENCES entities(id)
);

CREATE INDEX face_detections_embed_idx ON face_detections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
CREATE INDEX face_detections_entity_idx ON face_detections(entity_id) WHERE entity_id IS NOT NULL;
CREATE INDEX face_detections_clip_idx ON face_detections(clip_id);

-- Temporal state graph: per (entity, clip) state snapshot.
CREATE TABLE entity_states (
    id              bigserial PRIMARY KEY,
    entity_id       uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    clip_id         uuid NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    emotional_state text,
    knowledge       text,
    allegiance      text,
    goal            text,
    raw             jsonb NOT NULL,
    UNIQUE (entity_id, clip_id)
);

CREATE INDEX entity_states_clip_idx ON entity_states(clip_id);
CREATE INDEX entity_states_entity_idx ON entity_states(entity_id);

-- Relationship edges between entities.
CREATE TABLE entity_relations (
    id         bigserial PRIMARY KEY,
    video_id   uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    subject_id uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    object_id  uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate  text NOT NULL,
    clip_id    uuid REFERENCES clips(id) ON DELETE SET NULL,
    raw        jsonb NOT NULL
);

CREATE INDEX entity_relations_subject_idx ON entity_relations(subject_id);
CREATE INDEX entity_relations_object_idx ON entity_relations(object_id);

-- Background job queue. Consumed via SELECT ... FOR UPDATE SKIP LOCKED.
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

-- +goose Down

DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS entity_relations;
DROP TABLE IF EXISTS entity_states;
DROP TABLE IF EXISTS face_detections;
DROP TABLE IF EXISTS entities;
DROP TABLE IF EXISTS transcript_segments;
DROP TABLE IF EXISTS clips;
DROP TABLE IF EXISTS videos;

DROP EXTENSION IF EXISTS "pg_trgm";
DROP EXTENSION IF EXISTS "vector";
DROP EXTENSION IF EXISTS "pgcrypto";
