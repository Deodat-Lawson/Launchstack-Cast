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

    # --- Optional Gemini per-detection captioning -------------------------
    use_gemini = caption and _gemini_available()
    if caption and not use_gemini:
        typer.echo("GEMINI_API_KEY not set; skipping captions (use --no-caption to silence).")

    if use_gemini and crops:
        from drone_search.llm import GeminiClient

        # Untracked detections (track_id == -1) get a unique key per index so
        # each is captioned individually — they may be different people that
        # ByteTrack failed to associate.
        def _key(i: int, d: docmod.Document) -> int:
            return d.track_id if d.track_id != -1 else -(i + 1)

        rep_idx_by_key: dict[int, int] = {}
        for i, d in enumerate(detections):
            k = _key(i, d)
            cur = rep_idx_by_key.get(k)
            if cur is None or detections[cur].det_conf < d.det_conf:
                rep_idx_by_key[k] = i

        rep_indices = list(rep_idx_by_key.values())
        rep_crops = [crops[i] for i in rep_indices]
        typer.echo(
            f"captioning {len(rep_crops)} unique tracks with Gemini "
            f"(deduped from {len(crops)} detections) ..."
        )
        client = GeminiClient()
        rep_results = client.caption_crops(rep_crops)
        cap_by_key = {_key(i, detections[i]): r for i, r in zip(rep_indices, rep_results, strict=True)}

        for i, d in enumerate(detections):
            r = cap_by_key.get(_key(i, d))
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
