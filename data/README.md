# data/

Layout:
- `raw/` — Raw video clips (gitignored; multi-GB).
- `features/` — Cached YOLOv8+CLIP feature parquets per clip.
- `splits/` — Frozen test/dev splits (committed once at week 2 with SHAs).

The whole `data/` tree is gitignored except for this README and `.gitkeep`.
