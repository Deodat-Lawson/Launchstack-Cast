# Cast — Frontend prototype

Single-file React prototype of the Cast UI, delivered as a Claude Design handoff. The bundle landed here verbatim — treat it as a spec, not as code that is expected to scale. A production frontend (build tooling, TypeScript, real API integration) is still deferred until the backend pipeline lands real endpoints.

## Running

The API serves this directory as static files at the root URL. Start the stack the usual way:

```bash
make db-up && make migrate-up
make run-api
```

Then open <http://localhost:8080/>. `/` redirects to `/Cast.html` (the design ships under that name). The serve directory is configurable via `CAST_WEB_DIR` (default `./web`).

## What's here

- `Cast.html` — shell: fonts, CSS variables, React UMD + Babel standalone, root element.
- `app_simple.jsx` — the entire UI: upload landing, processing screen, analysis view (video card + scene timeline + per-character state timeline + cast list + search + results with reason breakdown).
- `data.js` — the fake corpus (video "The Cartographer's Apprentice" with 18 scenes and 5 characters). All retrieval is client-side against this.

No build step. Babel Standalone compiles the JSX in the browser, which is slow on first load but fine for a prototype.

## Wiring to real data (future)

Replace `data.js` with fetches against the Go API. This needs endpoints that don't exist yet:

- `GET /v1/videos` — library listing (replaces `VIDEO`).
- `GET /v1/videos/{id}/clips` — scenes + transcripts + presence + states (replaces `CLIPS`).
- `GET /v1/videos/{id}/entities` — cast list with face crops (replaces `ENTITIES`).
- `POST /v1/search` — hybrid retrieval with RRF breakdown (replaces `runQuery`).

Until these land, the prototype is a visual spec. Don't bolt half-real data in — either keep the whole prototype self-contained or do the backend work first.

## Not implemented from the original bundle

The handoff also contained an older dense terminal-style version under `components/` (`app.jsx`, `entity_rail.jsx`, `center_pane.jsx`, `right_pane.jsx`, `primitives.jsx`). The chat transcript shows the user rejected it for being too complex; `app_simple.jsx` is the replacement. Only the simplified version ships here.
