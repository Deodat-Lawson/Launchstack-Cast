"""DroneSearch — Streamlit app.

Tabs:
  Upload        — upload an mp4, run ingest in the background.
  Watch & Ask   — video player. Pause to summarize the moment or ask Gemini.
  Search        — corpus-wide hybrid retrieval (CLIP + inverted index + Rocchio).
  Inspect       — raw Documents browser, useful for debugging ingest output.

State management uses `st.session_state` keyed by upload sha1 so multiple
ingested videos coexist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Make the app/ helper module importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent))

from lib import (  # noqa: E402  (after sys.path manipulation)
    Upload,
    crop_at,
    frame_at,
    load_manifest,
    now_iso,
    run_ingest,
    save_manifest,
    store_upload,
)

from drone_search import retrieve  # noqa: E402
from drone_search.config import Paths  # noqa: E402
from drone_search.document import (  # noqa: E402
    Document,
    FrameCaption,
    frames_from_parquet,
    from_parquet,
)
from drone_search.embed import encode_text  # noqa: E402
from drone_search.index import dense as dense_idx  # noqa: E402
from drone_search.index.inverted import build_index as build_inverted  # noqa: E402

try:
    from streamlit_player import st_player  # type: ignore
    _HAS_PLAYER = True
except ImportError:
    _HAS_PLAYER = False


# ----- caches ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_docs(parquet_path: str, mtime: float) -> list[Document]:  # noqa: ARG001
    return from_parquet(parquet_path)


@st.cache_data(show_spinner=False)
def _load_frames(parquet_path: str, mtime: float) -> list[FrameCaption]:  # noqa: ARG001
    return frames_from_parquet(parquet_path)


@st.cache_resource(show_spinner=False)
def _build_dense(parquet_path: str, mtime: float):  # noqa: ARG001
    docs = _load_docs(parquet_path, mtime)
    embs = np.stack(
        [d.embedding for d in docs if d.embedding is not None],
        axis=0,
    ) if docs else np.zeros((0, 512), dtype=np.float32)
    return dense_idx.build_index(embs)


@st.cache_resource(show_spinner=False)
def _build_scene_dense(parquet_path: str, mtime: float):  # noqa: ARG001
    frames = _load_frames(parquet_path, mtime)
    embs = np.stack(
        [f.scene_embedding for f in frames if f.scene_embedding is not None],
        axis=0,
    ) if frames else np.zeros((0, 512), dtype=np.float32)
    return dense_idx.build_index(embs)


@st.cache_resource(show_spinner=False)
def _build_inverted(parquet_path: str, mtime: float):  # noqa: ARG001
    return build_inverted(_load_docs(parquet_path, mtime))


def _docs(upload: Upload) -> list[Document]:
    return _load_docs(str(upload.docs_parquet), upload.docs_parquet.stat().st_mtime)


def _frames(upload: Upload) -> list[FrameCaption]:
    return _load_frames(str(upload.frames_parquet), upload.frames_parquet.stat().st_mtime)


# ----- Gemini availability --------------------------------------------------


def _gemini_client():
    """Lazy singleton; returns None when GEMINI_API_KEY is unset."""
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return None
    if "_gemini" not in st.session_state:
        from drone_search.llm import GeminiClient

        st.session_state._gemini = GeminiClient()
    return st.session_state._gemini


# ----- Tabs -----------------------------------------------------------------


def upload_tab() -> None:
    st.subheader("Upload a clip")
    paths = Paths.from_env()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    max_dur = int(os.environ.get("DRONE_SEARCH_MAX_CLIP_DURATION_SEC", "300"))
    st.caption(f"mp4 / mov / avi, up to {max_dur // 60} min recommended for the demo deployment.")

    uploaded = st.file_uploader("video file", type=["mp4", "mov", "avi"])
    use_caption = st.checkbox(
        "Caption with Gemini",
        value=_gemini_client() is not None,
        help="Per-detection + scene captions. Disabled if GEMINI_API_KEY is unset.",
    )

    if not uploaded:
        return

    if st.button("Ingest"):
        sha, video_path = store_upload(uploaded.getvalue(), uploaded.name)
        out = paths.features / f"{sha}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)

        with st.status(f"Ingesting {uploaded.name} ...", expanded=True) as status:
            proc = run_ingest(video_path, out, caption=use_caption)
            assert proc.stdout is not None
            for line in proc.stdout:
                st.write(line.rstrip())
            ret = proc.wait()
            if ret != 0:
                status.update(label=f"Ingest failed (exit {ret})", state="error")
                return
            status.update(label="Ingest complete", state="complete")

        docs = _load_docs(str(out), out.stat().st_mtime)
        frames_path = out.with_name(out.stem + ".frames" + out.suffix)
        n_frames = len(_load_frames(str(frames_path), frames_path.stat().st_mtime)) if frames_path.exists() else 0

        manifest = load_manifest()
        manifest[sha] = Upload(
            sha1=sha,
            original_filename=uploaded.name,
            n_docs=len(docs),
            n_frames=n_frames,
            ingested_at=now_iso(),
        )
        save_manifest(manifest)
        st.session_state.selected_sha = sha
        st.success(f"Done. {len(docs)} detections, {n_frames} frames. Switch to Watch & Ask.")


def watch_tab(upload: Upload) -> None:
    docs = _docs(upload)
    frames = _frames(upload)

    if not frames:
        st.info("No frames in scene index — re-ingest with --caption to enable Watch & Ask.")
        return

    col_player, col_side = st.columns([2, 1])

    with col_player:
        seek_to = st.session_state.get("seek_to")
        st.video(str(upload.video_path), start_time=int(seek_to) if seek_to else 0)
        duration = frames[-1].t if frames else 60.0
        current_t = st.slider(
            "Current moment (s)",
            min_value=0.0,
            max_value=float(duration),
            value=float(seek_to or 0.0),
            step=0.1,
            key=f"slider-{upload.sha1}",
        )
        st.session_state.current_t = current_t

    nearest_frame = _nearest_frame(frames, current_t)
    detections_now = _detections_near(docs, current_t, window=0.6)

    with col_side:
        st.markdown(f"**At t = {current_t:.1f}s**")
        if nearest_frame and nearest_frame.scene_description:
            st.markdown(f"_Scene_: {nearest_frame.scene_description}")
        if detections_now:
            st.markdown(f"**Detections ({len(detections_now)})**")
            for d in detections_now[:6]:
                crop = crop_at(upload.video_path, d.t, d.bbox)
                if crop is not None:
                    st.image(crop, width=120, caption=f"track {d.track_id} — {d.description or '(no caption)'}")
                else:
                    st.write(f"track {d.track_id} (crop unavailable)")
        else:
            st.markdown("_No people detected at this moment._")

    st.divider()

    # --- Summarize / Ask -----------------------------------------------------
    client = _gemini_client()
    summary_col, ask_col = st.columns([1, 2])

    with summary_col:
        if st.button("Summarize this moment", disabled=client is None, use_container_width=True):
            if client is not None and nearest_frame is not None:
                key = (upload.sha1, nearest_frame.frame_id)
                cache = st.session_state.setdefault("moment_summary", {})
                if key not in cache:
                    img = frame_at(upload.video_path, current_t)
                    if img is not None:
                        ctx = _context_window(frames, nearest_frame, span=4)
                        cache[key] = client.summarize_moment(img, detections_now, ctx)
                    else:
                        cache[key] = "(could not extract frame)"
                st.session_state.last_summary = cache[key]
        if "last_summary" in st.session_state:
            st.markdown(st.session_state.last_summary)

    with ask_col:
        question = st.text_input("Ask about this moment", key=f"ask-{upload.sha1}")
        if st.button("Send", disabled=client is None or not question, use_container_width=True):
            img = frame_at(upload.video_path, current_t)
            if img is not None and client is not None:
                answer = client.answer_about_moment(question, img, detections_now)
                st.session_state.setdefault("qa_history", []).append(
                    {"t": current_t, "q": question, "a": answer}
                )

        for entry in reversed(st.session_state.get("qa_history", [])[-5:]):
            with st.container(border=True):
                st.markdown(f"**Q (t={entry['t']:.1f}s):** {entry['q']}")
                st.markdown(f"**A:** {entry['a']}")

    if client is None:
        st.caption("Gemini features disabled — set GEMINI_API_KEY to enable summarize / ask.")

    # --- Similar moments -----------------------------------------------------
    st.divider()
    st.markdown("**Similar moments in this video**")
    if nearest_frame is not None and nearest_frame.scene_embedding is not None:
        scene_index = _build_scene_dense(
            str(upload.frames_parquet), upload.frames_parquet.stat().st_mtime
        )
        scores, ids = dense_idx.search(scene_index, nearest_frame.scene_embedding[None, :], k=8)
        cols = st.columns(min(6, max(1, len(ids[0]))))
        col_i = 0
        for sc, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0 or idx >= len(frames):
                continue
            cand = frames[int(idx)]
            if cand.frame_id == nearest_frame.frame_id:
                continue
            with cols[col_i % len(cols)]:
                thumb = frame_at(upload.video_path, cand.t)
                if thumb is not None:
                    st.image(thumb, caption=f"t={cand.t:.1f}s — score {sc:.2f}")
                if st.button(f"Jump to t={cand.t:.1f}", key=f"jump-{cand.frame_id}"):
                    st.session_state.seek_to = cand.t
                    st.rerun()
            col_i += 1


def search_tab(upload: Upload) -> None:
    docs = _docs(upload)
    if not docs:
        st.info("No detections in this upload.")
        return

    st.subheader("Search this corpus")
    query = st.text_input("Query (natural language)")
    k = st.slider("Top k", 1, 30, 10)
    use_parse = st.checkbox(
        "Parse query with Gemini",
        value=_gemini_client() is not None,
        help="Extracts attribute tags from the query for the inverted index.",
    )
    use_summary = st.checkbox(
        "Summarize results with Gemini",
        value=_gemini_client() is not None,
    )
    alpha = st.slider("alpha (dense weight)", 0.0, 1.0, 0.7)
    beta = st.slider("beta (tag weight)", 0.0, 1.0, 0.3)

    if not query or not st.button("Search"):
        return

    parsed_tags: list[str] = []
    clip_phrase = query
    client = _gemini_client()
    if use_parse and client is not None:
        parsed = client.parse_query(query)
        clip_phrase = parsed.clip_phrase
        parsed_tags = parsed.attribute_tags
        st.caption(f"Parsed → phrase=`{clip_phrase}`, tags={parsed_tags or '[]'}")

    q_emb = encode_text([clip_phrase])[0]
    dense_index = _build_dense(str(upload.docs_parquet), upload.docs_parquet.stat().st_mtime)
    inv_index = _build_inverted(str(upload.docs_parquet), upload.docs_parquet.stat().st_mtime)

    hits = retrieve.hybrid_score(
        q_emb,
        parsed_tags,
        docs,
        dense_index,
        inv_index,
        alpha=alpha,
        beta=beta,
        k=k,
    )

    st.session_state.last_query_emb = q_emb
    st.session_state.last_hits = hits

    _render_hits(upload, docs, hits, query=query, summarize=use_summary)
    _render_feedback(upload, docs, hits, q_emb, parsed_tags, dense_index, inv_index, alpha, beta, k)


def _render_hits(
    upload: Upload,
    docs: list[Document],
    hits: list,
    *,
    query: str,
    summarize: bool,
) -> None:
    if not hits:
        st.info("No hits.")
        return

    if summarize and (client := _gemini_client()) is not None:
        thumbs: list[tuple[Document, Image.Image]] = []
        for h in hits[:5]:
            d = docs[h.doc_idx]
            crop = crop_at(upload.video_path, d.t, d.bbox)
            if crop is not None:
                thumbs.append((d, crop))
        if thumbs:
            st.markdown(f"**Summary:** {client.summarize_hits(query, thumbs)}")

    fb = st.session_state.setdefault("feedback", {})
    cols = st.columns(min(4, len(hits)))
    for i, h in enumerate(hits):
        d = docs[h.doc_idx]
        with cols[i % len(cols)]:
            crop = crop_at(upload.video_path, d.t, d.bbox)
            if crop is not None:
                st.image(crop, caption=f"t={d.t:.1f}s — {h.score:.2f}")
            st.caption(f"track {d.track_id}")
            if d.description:
                st.caption(d.description)
            mark = fb.get(h.doc_idx, 0)
            r1, r2 = st.columns(2)
            if r1.button("👍", key=f"up-{h.doc_idx}-{i}"):
                fb[h.doc_idx] = 1
            if r2.button("👎", key=f"down-{h.doc_idx}-{i}"):
                fb[h.doc_idx] = -1
            if mark == 1:
                st.caption("relevant")
            elif mark == -1:
                st.caption("irrelevant")


def _render_feedback(
    upload: Upload,
    docs: list[Document],
    hits: list,
    q_emb: np.ndarray,
    parsed_tags: list[str],
    dense_index,
    inv_index,
    alpha: float,
    beta: float,
    k: int,
) -> None:
    fb = st.session_state.get("feedback", {})
    if not fb:
        return

    if not st.button("Apply Rocchio feedback"):
        return

    rel = np.stack(
        [docs[i].embedding for i, m in fb.items() if m == 1 and docs[i].embedding is not None],
        axis=0,
    ) if any(m == 1 for m in fb.values()) else np.zeros((0, 512), dtype=np.float32)
    irr = np.stack(
        [docs[i].embedding for i, m in fb.items() if m == -1 and docs[i].embedding is not None],
        axis=0,
    ) if any(m == -1 for m in fb.values()) else np.zeros((0, 512), dtype=np.float32)

    new_q = retrieve.rocchio(q_emb, rel, irr)
    new_hits = retrieve.hybrid_score(
        new_q, parsed_tags, docs, dense_index, inv_index,
        alpha=alpha, beta=beta, k=k,
    )
    st.markdown("**Re-ranked after feedback**")
    _render_hits(upload, docs, new_hits, query="(re-ranked)", summarize=False)


def inspect_tab(upload: Upload) -> None:
    docs = _docs(upload)
    st.write(f"**{len(docs)}** Documents in `{upload.docs_parquet.name}`")
    if not docs:
        return

    n_show = st.slider("rows", 5, min(200, len(docs)), 20)
    rows = []
    for d in docs[:n_show]:
        rows.append(
            {
                "frame_id": d.frame_id,
                "t": round(d.t, 2),
                "track_id": d.track_id,
                "det_conf": round(d.det_conf, 3),
                "tags": ", ".join(d.tags),
                "description": d.description[:60] + ("…" if len(d.description) > 60 else ""),
            }
        )
    st.dataframe(rows, use_container_width=True)


# ----- helpers --------------------------------------------------------------


def _nearest_frame(frames: list[FrameCaption], t: float) -> FrameCaption | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f.t - t))


def _detections_near(docs: list[Document], t: float, *, window: float) -> list[Document]:
    return [d for d in docs if abs(d.t - t) < window]


def _context_window(frames: list[FrameCaption], anchor: FrameCaption, *, span: int) -> list[FrameCaption]:
    """`span` frames on either side of `anchor`, in time order."""
    sorted_by_t = sorted(frames, key=lambda f: f.t)
    try:
        i = sorted_by_t.index(anchor)
    except ValueError:
        return []
    lo = max(0, i - span)
    hi = min(len(sorted_by_t), i + span + 1)
    return sorted_by_t[lo:hi]


# ----- main -----------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="DroneSearch", layout="wide")
    st.title("DroneSearch")
    st.caption(
        "Drone-video person retrieval IR pipeline. "
        "Upload a clip, scrub to a moment, ask Gemini, or run a corpus search."
    )

    manifest = load_manifest()
    sha_options = {f"{u.original_filename} ({sha[:8]})": sha for sha, u in manifest.items()}
    if sha_options:
        default_label = next(iter(sha_options))
        st.sidebar.markdown("**Active upload**")
        chosen = st.sidebar.selectbox("Pick a video", list(sha_options.keys()), index=0, label_visibility="collapsed")
        st.session_state.selected_sha = sha_options[chosen]
        u = manifest[st.session_state.selected_sha]
        st.sidebar.caption(f"{u.n_docs} detections · {u.n_frames} frames\ningested {u.ingested_at}")
    else:
        st.sidebar.markdown("_no uploads yet — start in the Upload tab_")
        st.session_state.selected_sha = None

    st.sidebar.divider()
    st.sidebar.markdown(f"Gemini: {'✅ enabled' if _gemini_client() else '⚠️ no API key'}")
    st.sidebar.markdown(f"Player: {'streamlit-player' if _HAS_PLAYER else 'fallback (slider)'}")

    tabs = st.tabs(["Upload", "Watch & Ask", "Search", "Inspect"])
    with tabs[0]:
        upload_tab()

    sha = st.session_state.get("selected_sha")
    upload = manifest.get(sha) if sha else None

    with tabs[1]:
        if upload is None:
            st.info("No upload selected. Use the Upload tab first.")
        else:
            watch_tab(upload)
    with tabs[2]:
        if upload is None:
            st.info("No upload selected.")
        else:
            search_tab(upload)
    with tabs[3]:
        if upload is None:
            st.info("No upload selected.")
        else:
            inspect_tab(upload)


if __name__ == "__main__":
    main()
