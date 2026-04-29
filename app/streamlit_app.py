"""DroneSearch — Streamlit UI (week-1 placeholder).

Loads a parquet of Documents (output of `python -m drone_search ingest`) and
displays a metadata grid. Retrieval, RF, and alert log come in later weeks.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from drone_search.config import Paths
from drone_search.document import from_parquet


def main() -> None:
    st.set_page_config(page_title="DroneSearch", layout="wide")
    st.title("DroneSearch — feature browser")
    st.caption("Week-1 placeholder. Loads a Documents parquet and shows metadata.")

    paths = Paths.from_env()
    default = str(paths.features) if paths.features.exists() else str(paths.data_dir)
    parquet_path = st.sidebar.text_input("Documents parquet path", value=default)

    if not parquet_path:
        st.info("Enter a parquet path in the sidebar.")
        return

    p = Path(parquet_path)
    if not p.exists() or p.is_dir():
        st.warning(f"not a file: {parquet_path}")
        return

    docs = from_parquet(p)
    st.write(f"**{len(docs)}** Documents loaded from `{p}`")
    if not docs:
        return

    n_show = st.sidebar.slider("rows to show", 5, min(200, len(docs)), 20)
    rows = []
    for d in docs[:n_show]:
        rows.append(
            {
                "frame_id": d.frame_id,
                "t": round(d.t, 2),
                "track_id": d.track_id,
                "det_conf": round(d.det_conf, 3),
                "bbox": d.bbox,
                "tags": ", ".join(d.tags),
                "embed_dim": (d.embedding.shape[0] if d.embedding is not None else None),
            }
        )
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
