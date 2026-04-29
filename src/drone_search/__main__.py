"""DroneSearch CLI."""
from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

from drone_search import document as docmod
from drone_search.embed import encode_images
from drone_search.ingest import extract_detections, extract_frames

app = typer.Typer(add_completion=False, help="DroneSearch — drone person retrieval IR pipeline.")


@app.callback()
def _root() -> None:
    """Root callback — forces subcommand dispatch even when only one command exists."""


def _frames_parquet_path(out: Path) -> Path:
    """Sibling parquet for FrameCaption rows, derived from the detections out path."""
    return out.with_name(out.stem + ".frames" + out.suffix)


@app.command()
def ingest(
    video: Path = typer.Option(
        ..., "--video", "-v", exists=True, dir_okay=False, help="Path to a video file."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output parquet path for Documents."),
    fps: float = typer.Option(1.0, help="Frames per second to sample."),
    conf: float = typer.Option(0.25, help="YOLO confidence threshold."),
    max_frames: int | None = typer.Option(None, help="Cap on detections (debug)."),
    embed_batch: int = typer.Option(32, help="CLIP encode batch size."),
    yolo_model: str = typer.Option(
        os.environ.get("DRONE_SEARCH_YOLO_MODEL", "yolov8x.pt"),
        help="YOLO weights (yolov8n.pt for CPU deploys, yolov8x.pt for evaluation).",
    ),
    imgsz: int = typer.Option(
        int(os.environ.get("DRONE_SEARCH_IMGSZ", "1280")),
        "--imgsz",
        help="YOLO inference resolution. 1280 lifts small-person recall (broadcast/drone); 640 is faster.",
    ),
    min_bbox_area_frac: float = typer.Option(
        float(os.environ.get("DRONE_SEARCH_MIN_BBOX_AREA_FRAC", "0.0")),
        "--min-bbox-area-frac",
        help="Drop boxes smaller than this fraction of frame area. Try 0.0001 for soccer/stadium crowd suppression.",
    ),
    tracker: str = typer.Option(
        os.environ.get("DRONE_SEARCH_TRACKER", "botsort.yaml"),
        "--tracker",
        help="Ultralytics tracker config. botsort.yaml (default) handles crowded scenes better; bytetrack.yaml is faster.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device: cpu, cuda, mps. Defaults to DRONE_SEARCH_DEVICE env or auto-detect.",
    ),
    caption: bool = typer.Option(
        True,
        "--caption/--no-caption",
        help="Use Gemini for per-detection and scene captions. Auto-disabled if GEMINI_API_KEY is unset.",
    ),
    max_frame_captions: int = typer.Option(
        int(os.environ.get("DRONE_SEARCH_MAX_FRAME_CAPTIONS", "60")),
        help="Cap on scene captions to control Gemini cost.",
    ),
    caption_samples_per_track: int = typer.Option(
        int(os.environ.get("DRONE_SEARCH_CAPTION_SAMPLES_PER_TRACK", "1")),
        help="Crops to caption per real track (top-N by det_conf). Higher = richer tags.",
    ),
    caption_max_untracked: int = typer.Option(
        int(os.environ.get("DRONE_SEARCH_CAPTION_MAX_UNTRACKED", "64")),
        help="Cap on captions for untracked detections (track_id=-1), top-N by confidence.",
    ),
    merge_tracks: bool = typer.Option(
        True,
        "--merge-tracks/--no-merge-tracks",
        help="Cluster track centroids (cosine DBSCAN) to link re-entries of the same person.",
    ),
    merge_eps: float = typer.Option(
        float(os.environ.get("DRONE_SEARCH_MERGE_EPS", "0.2")),
        help="Cosine-distance eps for track merging. Lower = stricter (fewer merges).",
    ),
) -> None:
    """Run YOLOv8+ByteTrack on `video`, embed crops with CLIP, write Documents to `out`.

    Also writes a sibling `*.frames.parquet` with one row per sampled frame
    containing scene-level CLIP embeddings and (optional) Gemini scene captions.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    from drone_search.config import resolve_device

    resolved_device = resolve_device(device)
    typer.echo(f"using device={resolved_device}")

    # --- Detection pass ---------------------------------------------------
    detections: list[docmod.Document] = []
    crops = []
    typer.echo(f"detecting persons in {video} ...")
    for det, crop in extract_detections(
        video, fps=fps, conf=conf, max_frames=max_frames,
        model_name=yolo_model, device=resolved_device,
        imgsz=imgsz, min_bbox_area_frac=min_bbox_area_frac,
        tracker=tracker,
    ):
        detections.append(
            docmod.Document(
                frame_id=det.frame_id,
                t=det.t,
                bbox=det.bbox,
                track_id=det.track_id,
                det_conf=det.det_conf,
            )
        )
        crops.append(crop)

    typer.echo(f"detected {len(detections)} persons; embedding ...")
    if detections:
        embs = encode_images(crops, batch_size=embed_batch, device=resolved_device)
        for d, e in zip(detections, embs, strict=True):
            d.embedding = e

    # --- Track merging (re-id same person across fragmented ByteTrack ids) -
    if merge_tracks and detections:
        from drone_search.cluster import merge_tracks as _merge_tracks
        n_groups, n_collapsed = _merge_tracks(detections, eps=merge_eps)
        if n_collapsed:
            n_tracks_after = len({d.track_id for d in detections if d.track_id != -1})
            typer.echo(
                f"merged {n_collapsed} fragmented track(s) across {n_groups} identity group(s) "
                f"→ {n_tracks_after} unique tracks (eps={merge_eps})"
            )

    # --- Optional Gemini per-detection captioning -------------------------
    use_gemini = caption and _gemini_available()
    if caption and not use_gemini:
        typer.echo("GEMINI_API_KEY not set; skipping captions (use --no-caption to silence).")

    if use_gemini and crops:
        from drone_search.llm import GeminiClient

        # Bucket detections: real tracks keyed by track_id; untracked (-1) lumped
        # into one bucket since we can't associate them and don't want one
        # caption per untracked detection blowing up cost.
        tracked: dict[int, list[int]] = {}
        untracked: list[int] = []
        for i, d in enumerate(detections):
            if d.track_id == -1:
                untracked.append(i)
            else:
                tracked.setdefault(d.track_id, []).append(i)

        # Top-N by det_conf per real track.
        n_per_track = max(1, caption_samples_per_track)
        rep_indices: list[int] = []
        for tid, idxs in tracked.items():
            idxs.sort(key=lambda i: detections[i].det_conf, reverse=True)
            rep_indices.extend(idxs[:n_per_track])

        # Cap untracked at top-K by confidence.
        untracked.sort(key=lambda i: detections[i].det_conf, reverse=True)
        untracked_capped = untracked[:max(0, caption_max_untracked)]
        rep_indices.extend(untracked_capped)

        rep_crops = [crops[i] for i in rep_indices]
        typer.echo(
            f"captioning {len(rep_crops)} reps with Gemini "
            f"({len(tracked)} tracks × ≤{n_per_track} + {len(untracked_capped)}/"
            f"{len(untracked)} untracked; from {len(crops)} detections) ..."
        )
        client = GeminiClient()
        rep_results = client.caption_crops(rep_crops)

        # Map captions back: each rep captions its own detection, plus broadcasts
        # to the rest of its track so siblings inherit the description/tags.
        cap_by_idx: dict[int, "object"] = dict(zip(rep_indices, rep_results, strict=True))
        cap_by_track: dict[int, "object"] = {}
        for i, r in cap_by_idx.items():
            tid = detections[i].track_id
            if tid != -1 and tid not in cap_by_track:
                cap_by_track[tid] = r

        for i, d in enumerate(detections):
            r = cap_by_idx.get(i) or (cap_by_track.get(d.track_id) if d.track_id != -1 else None)
            if r is None:
                continue
            if not d.tags:
                d.tags = list(r.tags)
            d.description = r.description

    docmod.to_parquet(detections, out)
    typer.echo(f"wrote {len(detections)} Documents to {out}")

    # --- Frame pass (full-frame embeddings + optional scene captions) -----
    frames_out = _frames_parquet_path(out)
    typer.echo(f"sampling frames at {fps} fps for scene index ...")
    frame_records = list(extract_frames(video, fps=fps))
    if not frame_records:
        typer.echo("no frames sampled; skipping scene index")
        return

    frame_imgs = [img for _, _, img in frame_records]
    typer.echo(f"embedding {len(frame_imgs)} full frames ...")
    scene_embs = encode_images(frame_imgs, batch_size=embed_batch, device=resolved_device)

    captions: list[docmod.FrameCaption] = []
    if use_gemini:
        cap_n = min(len(frame_records), max_frame_captions)
        if cap_n < len(frame_records):
            typer.echo(
                f"capping scene captions at {cap_n} (of {len(frame_records)}); "
                f"raise DRONE_SEARCH_MAX_FRAME_CAPTIONS to caption more."
            )
        from drone_search.llm import GeminiClient

        client = GeminiClient()
        head = frame_records[:cap_n]
        captions = client.caption_frames(head)
        # Pad uncaptioned tail with empty FrameCaption rows so embedding rows still have a home.
        for fid, t, _ in frame_records[cap_n:]:
            captions.append(docmod.FrameCaption(frame_id=fid, t=t))
    else:
        for fid, t, _ in frame_records:
            captions.append(docmod.FrameCaption(frame_id=fid, t=t))

    for fc, emb in zip(captions, scene_embs, strict=True):
        fc.scene_embedding = emb

    docmod.frames_to_parquet(captions, frames_out)
    typer.echo(f"wrote {len(captions)} FrameCaptions to {frames_out}")


def _gemini_available() -> bool:
    """Cheap availability check that doesn't import the SDK."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


if __name__ == "__main__":
    app()
