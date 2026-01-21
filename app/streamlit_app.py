from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from stvis import pv_static

from build_vectors import build_song_vectors
from graph import build_pyvis_graph
from recommend import build_similarity_cache, get_topk_similar


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "split_song_lyrics_with_BERT2_emotions_en_100k.csv"
VECTORS_PATH = BASE_DIR / "artifacts" / "song_vectors.parquet"
SIM_CACHE_PATH = BASE_DIR / "artifacts" / "similarity_topk.parquet"


@st.cache_data(show_spinner=False)
def load_song_vectors(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_similarity_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def ensure_vectors() -> None:
    if VECTORS_PATH.exists():
        return
    if not DATA_PATH.exists():
        st.error(f"Missing input CSV: {DATA_PATH}")
        st.stop()
    with st.spinner("Building song vectors..."):
        build_song_vectors(DATA_PATH, VECTORS_PATH)
        load_song_vectors.clear()


def get_song_row(song_index: pd.DataFrame, song_key: str) -> pd.Series:
    row = song_index.loc[song_key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def format_labels(song_vectors: pd.DataFrame) -> pd.Series:
    base = song_vectors["song_key"].astype(str)

    if "title" in song_vectors.columns:
        title = song_vectors["title"].fillna("").astype(str)
        base = title.where(title != "", base)

    if "artist" in song_vectors.columns:
        artist = song_vectors["artist"].fillna("").astype(str)
        suffix = np.where(artist != "", " - " + artist, "")
        base = base + suffix

    base = base + " [" + song_vectors["song_key"].astype(str) + "]"
    return base


def build_extra_edges(
    song_vectors: pd.DataFrame,
    source_keys: list[str],
    expand_k: int,
    similarity_cache: pd.DataFrame | None,
    query_key: str,
) -> pd.DataFrame:
    rows = []
    for source_key in source_keys:
        source_key = str(source_key)
        neighbors = get_topk_similar(
            song_vectors, source_key, expand_k, similarity_cache=similarity_cache
        )
        for row in neighbors.itertuples(index=False):
            target_key = str(row.song_key)
            if target_key == query_key:
                continue
            rows.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "similarity": float(row.similarity),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source_key", "target_key", "similarity"])
    return pd.DataFrame(rows).drop_duplicates()


st.set_page_config(page_title="Emotion Similarity Explorer", layout="wide")

st.title("Emotion Similarity Explorer")
st.caption("Mean pooled GoEmotions vectors with cosine similarity")

ensure_vectors()

song_vectors = load_song_vectors(str(VECTORS_PATH))
if song_vectors.empty:
    st.warning("No song vectors found.")
    st.stop()

if "song_key" not in song_vectors.columns:
    st.error("song_vectors.parquet is missing 'song_key'.")
    st.stop()

song_vectors["song_key"] = song_vectors["song_key"].astype(str)

song_index = song_vectors.set_index("song_key", drop=False)
labels = format_labels(song_vectors)
label_map = dict(zip(song_vectors["song_key"].astype(str), labels))

st.sidebar.header("Search")
search_term = st.sidebar.text_input("Filter by title or artist")

filtered = song_vectors
if search_term:
    mask = pd.Series(False, index=song_vectors.index)
    lowered = search_term.lower()
    if "title" in song_vectors.columns:
        mask |= song_vectors["title"].fillna("").str.lower().str.contains(lowered)
    if "artist" in song_vectors.columns:
        mask |= song_vectors["artist"].fillna("").str.lower().str.contains(lowered)
    if "title" not in song_vectors.columns and "artist" not in song_vectors.columns:
        mask |= song_vectors["song_key"].astype(str).str.lower().str.contains(lowered)
    filtered = song_vectors[mask]

if filtered.empty:
    st.warning("No songs match the current filter.")
    st.stop()

filtered_keys = filtered["song_key"].astype(str).tolist()
selected_key = st.sidebar.selectbox(
    "Select a song",
    filtered_keys,
    format_func=lambda key: label_map.get(key, key),
)

max_k = max(1, min(50, len(song_vectors) - 1))
selected_k = st.sidebar.slider("Top-K", min_value=1, max_value=max_k, value=min(10, max_k))

expand_graph = st.sidebar.checkbox("Expand 2-hop neighbors", value=False)
expand_k = 3
if expand_graph:
    expand_k = st.sidebar.slider("2-hop per neighbor", min_value=1, max_value=10, value=3)

similarity_threshold = st.sidebar.slider(
    "Min similarity (graph)", min_value=0.0, max_value=1.0, value=0.0, step=0.01
)

cache_df = None
if SIM_CACHE_PATH.exists():
    cache_df = load_similarity_cache(str(SIM_CACHE_PATH))

if st.sidebar.button("Rebuild song vectors"):
    with st.spinner("Rebuilding song vectors..."):
        build_song_vectors(DATA_PATH, VECTORS_PATH)
        load_song_vectors.clear()
    st.rerun()

if st.sidebar.button("Build similarity cache (current K)"):
    with st.spinner("Building similarity cache..."):
        try:
            cache_df = build_similarity_cache(song_vectors, selected_k, SIM_CACHE_PATH)
            load_similarity_cache.clear()
        except ValueError as exc:
            st.sidebar.error(str(exc))
    st.rerun()

query_row = get_song_row(song_index, selected_key)

st.subheader("Query song")

meta_lines = []
if "title" in query_row:
    meta_lines.append(f"Title: {query_row['title']}")
if "artist" in query_row:
    meta_lines.append(f"Artist: {query_row['artist']}")
if "year" in query_row and pd.notna(query_row["year"]):
    meta_lines.append(f"Year: {query_row['year']}")
meta_lines.append(f"Song key: {selected_key}")

st.write(" | ".join(meta_lines))

neighbors = get_topk_similar(song_vectors, selected_key, selected_k, similarity_cache=cache_df)
if not neighbors.empty:
    neighbors["song_key"] = neighbors["song_key"].astype(str)

st.subheader("Top-K similar songs")
if neighbors.empty:
    st.info("No neighbors found.")
else:
    st.dataframe(neighbors, use_container_width=True)

extra_edges = None
if expand_graph and not neighbors.empty:
    max_sources = min(len(neighbors), 5)
    source_keys = neighbors["song_key"].astype(str).head(max_sources).tolist()
    extra_edges = build_extra_edges(
        song_vectors,
        source_keys,
        expand_k,
        similarity_cache=cache_df,
        query_key=selected_key,
    )

st.subheader("Connected graph")
if neighbors.empty:
    st.info("Select a song with available neighbors to render the graph.")
else:
    net = build_pyvis_graph(
        song_index,
        selected_key,
        neighbors,
        extra_edges=extra_edges,
        similarity_threshold=similarity_threshold if similarity_threshold > 0 else None,
    )
    pv_static(net)
