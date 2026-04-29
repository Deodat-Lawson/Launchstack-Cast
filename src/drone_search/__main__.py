"""DroneSearch CLI."""
from __future__ import annotations

from pathlib import Path

import typer

from drone_search import document as docmod
from drone_search.embed import encode_images
from drone_search.ingest import extract_detections

app = typer.Typer(add_completion=False, help="DroneSearch — drone person retrieval IR pipeline.")


@app.callback()
def _root() -> None:
    """Root callback — forces subcommand dispatch even when only one command exists."""


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
) -> None:
    """Run YOLOv8+ByteTrack on `video`, embed crops with CLIP, write Documents to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)

    detections: list[docmod.Document] = []
    crops = []
    typer.echo(f"detecting persons in {video} ...")
    for det, crop in extract_detections(video, fps=fps, conf=conf, max_frames=max_frames):
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
        embs = encode_images(crops, batch_size=embed_batch)
        for d, e in zip(detections, embs, strict=True):
            d.embedding = e

    docmod.to_parquet(detections, out)
    typer.echo(f"wrote {len(detections)} Documents to {out}")


if __name__ == "__main__":
    app()
