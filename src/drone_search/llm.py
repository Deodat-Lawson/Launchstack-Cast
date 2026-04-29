"""Gemini-3-Pro understanding layer.

Single place that touches the Gemini SDK. Six call sites, all funnel through
`GeminiClient.generate(...)` so retries, JSON parsing, and structured-output
enforcement live together.

Maps to proposal §11 stretch: Option-10 LLM-mediated query + scene captioning.
The IR core (CLIP+FAISS+inverted index, Rocchio, Bayesian) is not replaced —
this layer wraps it.
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from drone_search.document import Document, FrameCaption

log = logging.getLogger(__name__)

_MODEL = os.environ.get("DRONE_SEARCH_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

# Free-tier flash-lite is 15 RPM; 4s/call leaves a small margin.
_MIN_INTERVAL_S = float(os.environ.get("DRONE_SEARCH_GEMINI_MIN_INTERVAL_S", "4.0"))
_MAX_RETRIES = int(os.environ.get("DRONE_SEARCH_GEMINI_MAX_RETRIES", "4"))


def _parse_retry_delay(details: Any) -> float | None:
    """Extract `retryDelay` (seconds) from a Gemini APIError payload, if present."""
    try:
        items = details.get("error", {}).get("details", []) if isinstance(details, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "RetryInfo" in str(item.get("@type", "")):
                raw = str(item.get("retryDelay", "")).strip()
                if raw.endswith("s"):
                    raw = raw[:-1]
                return float(raw) if raw else None
    except (ValueError, TypeError, AttributeError):
        return None
    return None


@dataclass(slots=True)
class CaptionResult:
    description: str = ""
    tags: list[str] = field(default_factory=list)
    salient_features: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedQuery:
    clip_phrase: str
    attribute_tags: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


class GeminiClient:
    """Thin wrapper around google.genai. Constructed once per process.

    Raises `RuntimeError` at construction if `GEMINI_API_KEY` is not set.
    Callers should check `is_available()` first if the key is optional.
    """

    def __init__(self, api_key: str | None = None, model: str = _MODEL) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._last_call_t: float = 0.0

    def _throttle(self) -> None:
        wait = _MIN_INTERVAL_S - (time.monotonic() - self._last_call_t)
        if wait > 0:
            time.sleep(wait)
        self._last_call_t = time.monotonic()

    @staticmethod
    def is_available() -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    # --- core call ---------------------------------------------------------

    def generate(
        self,
        parts: list[Any],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Send `parts` (text + PIL images) to Gemini, return raw response text.

        When `json_schema` is provided, structured JSON output is enforced and
        the returned text is guaranteed to parse.
        """
        from google.genai import types

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = json_schema

        contents = [_to_part(p) for p in parts]
        cfg = types.GenerateContentConfig(**config) if config else None

        from google.genai import errors as genai_errors

        retryable = {429, 500, 502, 503, 504}
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=cfg,
                )
                return resp.text or ""
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                if code not in retryable or attempt == _MAX_RETRIES:
                    raise
                delay = _parse_retry_delay(getattr(e, "details", None))
                # Fall back to exponential backoff with jitter if server didn't say.
                # 503 (overload) deserves a longer floor than 429 (rate limit).
                if delay is None or delay <= 0:
                    base = 5.0 if code == 503 else 2.0
                    delay = min(60.0, base * (2.0 ** attempt)) + random.uniform(0, 1)
                log.info(
                    "Gemini %s; sleeping %.1fs (attempt %d/%d)",
                    code, delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)
        return ""  # unreachable

    # --- A2: per-detection captioning -------------------------------------

    def caption_crops(self, crops: list[Image.Image], *, batch_size: int = 16) -> list[CaptionResult]:
        """Return one CaptionResult per crop, in order."""
        if not crops:
            return []

        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "salient_features": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["description", "tags"],
                    },
                }
            },
            "required": ["results"],
        }

        out: list[CaptionResult] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            prompt = (
                "You are tagging cropped images of individual people from drone aerial footage. "
                f"For each of the {len(batch)} images below, in order, return a JSON object "
                "{results: [{description, tags, salient_features}]} where:\n"
                "- description: one short sentence describing the person\n"
                "- tags: lower_snake_case attributes (e.g. red_jacket, walking, carrying_backpack)\n"
                "- salient_features: 1-3 distinguishing visual cues for re-identification"
            )
            parts = [prompt, *batch]
            try:
                raw = self.generate(parts, json_schema=schema)
                data = json.loads(raw)
                results = data.get("results", [])
            except Exception as e:
                log.warning("caption_crops batch failed: %s", e)
                results = []

            for i in range(len(batch)):
                if i < len(results):
                    r = results[i]
                    out.append(
                        CaptionResult(
                            description=str(r.get("description", "")),
                            tags=[str(t) for t in r.get("tags", [])],
                            salient_features=[str(s) for s in r.get("salient_features", [])],
                        )
                    )
                else:
                    out.append(CaptionResult())
        return out

    # --- A3: per-frame scene captioning -----------------------------------

    def caption_frames(self, frames: list[tuple[int, float, Image.Image]]) -> list[FrameCaption]:
        """Scene-level captions for full frames. `frames` is [(frame_id, t, image), ...]."""
        if not frames:
            return []

        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "scene_tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["description", "scene_tags"],
                    },
                }
            },
            "required": ["results"],
        }

        out: list[FrameCaption] = []
        # Smaller batches for full frames since each image is bigger.
        batch_size = 4
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            prompt = (
                f"You are summarizing {len(batch)} frames from a drone aerial video, in order. "
                "For each frame return JSON {results: [{description, scene_tags}]}:\n"
                "- description: one or two sentences describing what is happening in the scene\n"
                "- scene_tags: lower_snake_case scene descriptors (e.g. urban_street, two_people, "
                "running, daytime)"
            )
            images = [img for (_, _, img) in batch]
            parts = [prompt, *images]
            try:
                raw = self.generate(parts, json_schema=schema)
                data = json.loads(raw)
                results = data.get("results", [])
            except Exception as e:
                log.warning("caption_frames batch failed: %s", e)
                results = []

            for i, (fid, t, _) in enumerate(batch):
                r = results[i] if i < len(results) else {}
                out.append(
                    FrameCaption(
                        frame_id=fid,
                        t=t,
                        scene_description=str(r.get("description", "")),
                        scene_tags=[str(x) for x in r.get("scene_tags", [])],
                    )
                )
        return out

    # --- A5: moment-centric calls -----------------------------------------

    def summarize_moment(
        self,
        frame: Image.Image,
        detections: list[Document],
        context_window: list[FrameCaption],
    ) -> str:
        det_lines = [
            f"- track {d.track_id}: {d.description or '(no caption)'} "
            f"[tags: {', '.join(d.tags) or 'none'}]"
            for d in detections
        ]
        ctx_lines = [
            f"- t={c.t:.1f}s: {c.scene_description or '(no caption)'}"
            for c in context_window
        ]
        prompt = (
            "You are explaining a single moment in a drone aerial video to a viewer who paused "
            "playback here. Use the current frame plus context to write one short paragraph "
            "(2-4 sentences) describing what is happening at this moment.\n\n"
            f"People detected at this moment:\n{chr(10).join(det_lines) or '(none)'}\n\n"
            f"Surrounding scene context:\n{chr(10).join(ctx_lines) or '(none)'}"
        )
        return self.generate([prompt, frame], max_tokens=512).strip()

    def answer_about_moment(
        self,
        question: str,
        frame: Image.Image,
        detections: list[Document],
    ) -> str:
        det_lines = [
            f"- track {d.track_id}: {d.description or '(no caption)'} "
            f"[tags: {', '.join(d.tags) or 'none'}]"
            for d in detections
        ]
        prompt = (
            "You are answering a viewer's question about a paused moment in a drone aerial video. "
            "Ground your answer strictly in the current frame and the listed detections. If the "
            "frame does not contain enough information, say so plainly.\n\n"
            f"People detected at this moment:\n{chr(10).join(det_lines) or '(none)'}\n\n"
            f"Question: {question}"
        )
        return self.generate([prompt, frame], max_tokens=512).strip()

    # --- A4: query parsing + summary --------------------------------------

    def parse_query(self, text: str) -> ParsedQuery:
        schema = {
            "type": "object",
            "properties": {
                "clip_phrase": {"type": "string"},
                "attribute_tags": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "object"},
            },
            "required": ["clip_phrase", "attribute_tags"],
        }
        prompt = (
            "Parse this drone-video search query into:\n"
            "- clip_phrase: a short visual phrase suitable for CLIP text encoding\n"
            "- attribute_tags: lower_snake_case tags matching the proposal vocab "
            "(e.g. red_jacket, walking, carrying_backpack)\n"
            "- filters: optional structured constraints like {min_track_id: int}\n\n"
            f"Query: {text}"
        )
        try:
            raw = self.generate([prompt], json_schema=schema, max_tokens=256)
            data = json.loads(raw)
            return ParsedQuery(
                clip_phrase=str(data.get("clip_phrase", text)),
                attribute_tags=[str(t) for t in data.get("attribute_tags", [])],
                filters=dict(data.get("filters", {})),
            )
        except Exception as e:
            log.warning("parse_query failed, falling back to raw text: %s", e)
            return ParsedQuery(clip_phrase=text)

    def summarize_hits(
        self,
        query: str,
        hits: list[tuple[Document, Image.Image]],
    ) -> str:
        if not hits:
            return ""
        lines = [
            f"- track {d.track_id} at t={d.t:.1f}s: {d.description or '(no caption)'} "
            f"[tags: {', '.join(d.tags) or 'none'}]"
            for d, _ in hits
        ]
        prompt = (
            f"A user searched for: '{query}'.\n"
            f"The retrieval system returned {len(hits)} top matches:\n"
            f"{chr(10).join(lines)}\n\n"
            "Write one short paragraph (2-4 sentences) summarizing the matches, focusing on "
            "what the user is most likely to care about. Reference timestamps where useful."
        )
        parts: list[Any] = [prompt]
        parts.extend(img for _, img in hits)
        return self.generate(parts, max_tokens=512).strip()


def _to_part(p: Any) -> Any:
    """Coerce a raw input (str | PIL.Image | google.genai.types.Part) into a Part."""
    from google.genai import types

    if isinstance(p, Image.Image):
        buf = io.BytesIO()
        p.convert("RGB").save(buf, format="JPEG", quality=88)
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    return p
