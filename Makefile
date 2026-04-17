.PHONY: dev run-api run-worker build test test-go test-python lint migrate-up migrate-down migrate-create db-up db-down clean e2e

# --- Config ---
DATABASE_URL ?= postgres://cast:cast@localhost:5432/cast?sslmode=disable
CAST_PYTHON_BIN ?= python3

# --- Dev ---
dev: db-up migrate-up run-api

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

# --- Build ---
build:
	go build -o bin/cast-api ./cmd/cast-api
	go build -o bin/cast-worker ./cmd/cast-worker

run-api:
	go run ./cmd/cast-api

run-worker:
	go run ./cmd/cast-worker

# --- Test ---
test: test-go test-python

test-go:
	go test ./... -count=1

test-python:
	cd python && $(CAST_PYTHON_BIN) -m pytest -x

lint:
	go vet ./...
	gofmt -l .

# --- Migrations (goose) ---
GOOSE := goose -dir migrations

migrate-up:
	$(GOOSE) postgres "$(DATABASE_URL)" up

migrate-down:
	$(GOOSE) postgres "$(DATABASE_URL)" down

migrate-create:
	@read -p "Migration name: " name; \
	$(GOOSE) create $$name sql -dir migrations

# --- Cleanup ---
clean:
	rm -rf bin/

# --- E2E (requires running services) ---
e2e:
	go test ./... -tags=e2e -count=1 -timeout=120s
