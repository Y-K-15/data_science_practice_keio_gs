from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st

from build_vectors import build_song_vectors
from graph import build_pyvis_graph
from recommend import (
    EMOTION_COLUMNS,
    build_annoy_index,
    build_similarity_cache,
    get_topk_similar,
    get_topk_similar_annoy,
)
from vis_component import vis_network


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "split_song_lyrics_with_BERT2_emotions_en_100k.csv"
VECTORS_PATH = BASE_DIR / "artifacts" / "song_vectors.parquet"
SIM_CACHE_PATH = BASE_DIR / "artifacts" / "similarity_topk.parquet"
ANNOY_TREES = 20
SHOW_REBUILD_VECTORS_BUTTON = False
SHOW_MIN_SIMILARITY_SLIDER = False
DETAILS_VIEW_QUERY_PARAM = "view"
DETAILS_VIEW_VALUE = "similarity_method_details"
DETAILS_MD_PATH = BASE_DIR / "docs" / "similarity_methods_detailed.md"


@st.cache_data(show_spinner=False)
def load_song_vectors(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_similarity_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_annoy_index(song_vectors: pd.DataFrame, n_trees: int):
    return build_annoy_index(song_vectors, n_trees=n_trees)


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

    return base


def parse_px(value: str, default: int) -> int:
    try:
        return int(str(value).replace("px", ""))
    except (TypeError, ValueError):
        return default


def get_component_ts(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    try:
        return int(value.get("ts") or 0)
    except (TypeError, ValueError):
        return 0


def _get_query_param(name: str) -> str:
    value = st.query_params.get(name)
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def render_similarity_method_details_page() -> None:
    st.title("Similarity method: 詳細説明")
    if st.button("アプリに戻る", type="secondary"):
        st.query_params.clear()
        st.rerun()

    if DETAILS_MD_PATH.exists():
        st.markdown(DETAILS_MD_PATH.read_text(encoding="utf-8"))
        return

    st.error(f"詳細説明ファイルが見つかりません: {DETAILS_MD_PATH}")


def options_to_dict(options) -> dict:
    if hasattr(options, "to_json"):
        return json.loads(options.to_json())
    if isinstance(options, str):
        return json.loads(options)
    if isinstance(options, dict):
        return options
    return {}


def build_extra_edges(
    song_vectors: pd.DataFrame,
    source_keys: list[str],
    expand_k: int,
    query_key: str,
    topk_func,
) -> pd.DataFrame:
    rows = []
    for source_key in source_keys:
        source_key = str(source_key)
        neighbors = topk_func(source_key, expand_k)
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


def empty_edges_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_key", "target_key", "similarity"])


SIM_METHODS = {
    "teammate_annoy": {
        "label": "Annoy angular",
        "desc": "Annoyのangular距離で近傍検索（cosine相当の近似検索）",
    },
    "raw": {
        "label": "raw cosine",
        "desc": "28次元をそのままL2正規化してcosineを計算",
    },
    "no_neutral": {
        "label": "cosine without neutral",
        "desc": "neutral列を除外してcosineを計算",
    },
    "mean_center": {
        "label": "mean-centered cosine",
        "desc": "全曲平均ベクトルを引いてからcosineを計算",
    },
    "zscore": {
        "label": "zscore cosine",
        "desc": "各列をz-score標準化してからcosineを計算（stdの下限を確保）",
    },
    "weighted": {
        "label": "weighted pooling (1-neutral)",
        "desc": "neutralが高い行ほど重みを下げるイメージで、(1-neutral)をスカラーで掛けてcosineを計算",
    },
}


@st.cache_data(show_spinner=False)
def prepare_matrix(song_vectors: pd.DataFrame, method: str) -> np.ndarray:
    df = song_vectors[EMOTION_COLUMNS].copy()
    method = method if method in SIM_METHODS else "raw"

    if method == "no_neutral" and "neutral" in df.columns:
        df = df.drop(columns=["neutral"])
    elif method == "mean_center":
        mu = df.mean()
        df = df - mu
    elif method == "zscore":
        mu = df.mean()
        std = df.std()
        std = std.replace(0, 1e-6)
        df = (df - mu) / std
    elif method == "weighted":
        if "neutral" in df.columns:
            w = 1.0 - df["neutral"].clip(lower=0.0, upper=1.0)
        else:
            w = 1.0
        df = df.mul(w, axis=0)

    return df.to_numpy(dtype=float)


def compute_topk_from_matrix(
    song_vectors: pd.DataFrame, matrix: np.ndarray, song_key: str, k: int
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be >= 1")

    if "song_key" not in song_vectors.columns:
        raise ValueError("song_vectors must include 'song_key'")

    keys = song_vectors["song_key"].astype(str).tolist()
    try:
        query_idx = keys.index(str(song_key))
    except ValueError:
        raise ValueError(f"song_key not found: {song_key}")

    # L2正規化
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    mat_norm = matrix / norms

    sims = mat_norm[[query_idx]] @ mat_norm.T
    sims = sims.flatten()
    sims[query_idx] = -np.inf

    max_k = min(k, len(sims) - 1) if len(sims) > 1 else 0
    if max_k <= 0:
        return pd.DataFrame(columns=["similarity", "song_key"])

    if max_k < len(sims):
        top_idx = np.argpartition(sims, -max_k)[-max_k:]
    else:
        top_idx = np.arange(len(sims))
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    meta_cols = [c for c in ["title", "artist", "year", "n_lines"] if c in song_vectors.columns]
    result = song_vectors.loc[top_idx, ["song_key"] + meta_cols].copy()
    result.insert(0, "similarity", sims[top_idx])
    return result.reset_index(drop=True)


def youtube_search_url(title: str, artist: str | None, song_key: str) -> str:
    safe_title = "" if title is None or pd.isna(title) else str(title).strip()
    safe_artist = "" if artist is None or pd.isna(artist) else str(artist).strip()
    query = safe_title if safe_title else str(song_key)
    if safe_artist:
        query = f"{query} {safe_artist}"
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


st.set_page_config(page_title="Emotion Similarity Explorer", layout="wide")

if _get_query_param(DETAILS_VIEW_QUERY_PARAM) == DETAILS_VIEW_VALUE:
    render_similarity_method_details_page()
    st.stop()

st.title("Emotion Similarity Explorer")

st.markdown(
    """
<style>
a.yt-btn {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  background: #ff0000;
  color: #ffffff !important;
  border-radius: 0.4rem;
  text-decoration: none !important;
  font-weight: 600;
  text-align: center;
  white-space: nowrap;
}
a.yt-btn:hover {
  background: #cc0000;
}
</style>
""",
    unsafe_allow_html=True,
)

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

method_keys = list(SIM_METHODS.keys())
default_method = "teammate_annoy" if "teammate_annoy" in method_keys else method_keys[0]
selected_method = st.sidebar.selectbox(
    "Similarity method",
    method_keys,
    format_func=lambda k: SIM_METHODS[k]["label"],
    index=method_keys.index(default_method),
)

if "show_method_help" not in st.session_state:
    st.session_state.show_method_help = False

if st.sidebar.button("説明を表示/非表示", use_container_width=True):
    st.session_state.show_method_help = not st.session_state.show_method_help

st.sidebar.link_button(
    "詳細説明",
    url=f"?{DETAILS_VIEW_QUERY_PARAM}={DETAILS_VIEW_VALUE}",
    help="新しいタブで詳細説明を開きます",
    type="secondary",
    use_container_width=True,
)

if st.session_state.show_method_help:
    st.sidebar.info(
        "\n".join(
            [
                f"- {SIM_METHODS[k]['label']}: {SIM_METHODS[k]['desc']}"
                for k in method_keys
            ]
        )
    )

st.sidebar.subheader("Graph controls")
expand_k = st.sidebar.slider("Expand neighbors per click", min_value=1, max_value=10, value=4)

similarity_threshold = 0.0
if SHOW_MIN_SIMILARITY_SLIDER:
    similarity_threshold = st.sidebar.slider(
        "Min similarity (graph)", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

cache_df = None
if SIM_CACHE_PATH.exists():
    cache_df = load_similarity_cache(str(SIM_CACHE_PATH))

if SHOW_REBUILD_VECTORS_BUTTON and st.sidebar.button("Rebuild song vectors"):
    with st.spinner("Rebuilding song vectors..."):
        build_song_vectors(DATA_PATH, VECTORS_PATH)
        load_song_vectors.clear()
    st.rerun()

if selected_method != "teammate_annoy":
    if st.sidebar.button("Build similarity cache (current K)"):
        with st.spinner("Building similarity cache..."):
            try:
                cache_df = build_similarity_cache(song_vectors, selected_k, SIM_CACHE_PATH)
                load_similarity_cache.clear()
            except ValueError as exc:
                st.sidebar.error(str(exc))
        st.rerun()

if "expanded_keys" not in st.session_state:
    st.session_state.expanded_keys = []
if "expanded_edges" not in st.session_state:
    st.session_state.expanded_edges = empty_edges_df()
if "focus_key" not in st.session_state:
    st.session_state.focus_key = selected_key
if "selected_node_key" not in st.session_state:
    st.session_state.selected_node_key = ""
if "last_click_ts" not in st.session_state:
    st.session_state.last_click_ts = 0
if "last_expand_k" not in st.session_state:
    st.session_state.last_expand_k = expand_k

if st.session_state.get("current_query_key") != selected_key:
    st.session_state.current_query_key = selected_key
    st.session_state.expanded_keys = []
    st.session_state.expanded_edges = empty_edges_df()
    st.session_state.focus_key = selected_key
    st.session_state.selected_node_key = ""
    st.session_state.last_click_ts = get_component_ts(st.session_state.get("connected_graph"))

if st.session_state.last_expand_k != expand_k:
    st.session_state.last_expand_k = expand_k
    st.session_state.expanded_keys = []
    st.session_state.expanded_edges = empty_edges_df()
    st.session_state.focus_key = selected_key
    st.session_state.selected_node_key = ""
    st.session_state.last_click_ts = get_component_ts(st.session_state.get("connected_graph"))

if st.sidebar.button("Reset graph", use_container_width=True):
    st.session_state.expanded_keys = []
    st.session_state.expanded_edges = empty_edges_df()
    st.session_state.focus_key = selected_key
    st.session_state.selected_node_key = ""
    st.session_state.last_click_ts = get_component_ts(st.session_state.get("connected_graph"))
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

st.write(" | ".join(meta_lines))
query_youtube_url = youtube_search_url(
    query_row.get("title", None),
    query_row.get("artist", None),
    selected_key,
)
st.markdown(
    f'<a class="yt-btn" href="{query_youtube_url}" target="_blank" rel="noopener noreferrer">YouTubeで見る</a>',
    unsafe_allow_html=True,
)

annoy_index = None
annoy_key_to_index = None
matrix_override = None

if selected_method == "teammate_annoy":
    annoy_index, annoy_key_to_index = load_annoy_index(song_vectors, ANNOY_TREES)
    neighbors = get_topk_similar_annoy(
        song_vectors, selected_key, selected_k, annoy_index, annoy_key_to_index
    )
elif selected_method == "raw":
    neighbors = get_topk_similar(
        song_vectors, selected_key, selected_k, similarity_cache=cache_df
    )
else:
    matrix_override = prepare_matrix(song_vectors, selected_method)
    neighbors = compute_topk_from_matrix(
        song_vectors, matrix_override, selected_key, selected_k
    )

if not neighbors.empty:
    neighbors["song_key"] = neighbors["song_key"].astype(str)

st.subheader("Top-K similar songs")
if neighbors.empty:
    st.info("No neighbors found.")
else:
    neighbors_view = neighbors.copy()
    if "title" in neighbors_view.columns:
        title_col = neighbors_view["title"]
    else:
        title_col = pd.Series([""] * len(neighbors_view))
    if "artist" in neighbors_view.columns:
        artist_col = neighbors_view["artist"]
    else:
        artist_col = pd.Series([None] * len(neighbors_view))
    neighbors_view.insert(
        len(neighbors_view.columns),
        "youtube",
        [
            youtube_search_url(t, a, k)
            for t, a, k in zip(title_col, artist_col, neighbors_view["song_key"])
        ],
    )
    neighbors_view = neighbors_view.drop(columns=["similarity", "song_key", "n_lines"], errors="ignore")

    st.markdown(
        """
<style>
[data-testid="stDataFrame"] a {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  background: #ff0000;
  color: #ffffff !important;
  border-radius: 0.4rem;
  text-decoration: none !important;
  font-weight: 600;
}
[data-testid="stDataFrame"] a:hover {
  background: #cc0000;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.dataframe(
        neighbors_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "youtube": st.column_config.LinkColumn(
                "YouTube",
                help="YouTube検索結果を開きます",
                display_text="YouTubeで見る",
            )
        },
    )

def _topk_func(song_key: str, k: int) -> pd.DataFrame:
    if selected_method == "teammate_annoy":
        if annoy_index is None or annoy_key_to_index is None:
            raise RuntimeError("Annoy index is not initialized.")
        return get_topk_similar_annoy(
            song_vectors, song_key, k, annoy_index, annoy_key_to_index
        )
    if selected_method == "raw":
        return get_topk_similar(song_vectors, song_key, k, similarity_cache=cache_df)
    mat = matrix_override if matrix_override is not None else prepare_matrix(
        song_vectors, selected_method
    )
    return compute_topk_from_matrix(song_vectors, mat, song_key, k)

st.subheader("Connected graph")
st.caption("ノードをクリックすると詳細が表示されます。「拡張する」で近傍を追加できます。")
if neighbors.empty:
    st.info("Select a song with available neighbors to render the graph.")
else:
    graph_state = st.session_state.get("connected_graph")
    if isinstance(graph_state, dict):
        clicked_key = str(graph_state.get("clicked", ""))
        click_ts = get_component_ts(graph_state)
        if clicked_key and click_ts > st.session_state.last_click_ts:
            st.session_state.last_click_ts = click_ts
            st.session_state.selected_node_key = clicked_key
            st.session_state.focus_key = clicked_key

    node_key = str(st.session_state.get("selected_node_key") or "")

    if node_key:
        try:
            card = st.container(border=True)
        except TypeError:
            card = st.container()

        with card:
            node_row = get_song_row(song_index, node_key) if node_key in song_index.index else None
            node_title = ""
            node_artist = ""
            if node_row is not None:
                node_title = (
                    "" if pd.isna(node_row.get("title", "")) else str(node_row.get("title", "")).strip()
                )
                node_artist = (
                    "" if pd.isna(node_row.get("artist", "")) else str(node_row.get("artist", "")).strip()
                )

            display_title = node_title if node_title else node_key
            st.markdown(f"**{display_title}**")
            if node_artist:
                st.write(f"Artist: {node_artist}")
            st.caption(f"song_key: {node_key}")

            youtube_url = youtube_search_url(node_title, node_artist, node_key)
            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown(
                    f'<a class="yt-btn" href="{youtube_url}" target="_blank" rel="noopener noreferrer">YouTubeで見る</a>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                can_expand = node_key != selected_key and node_key not in st.session_state.expanded_keys
                if st.button(
                    "拡張する",
                    disabled=not can_expand,
                    use_container_width=True,
                    key="expand_selected_node",
                ):
                    new_edges = build_extra_edges(
                        song_vectors,
                        [node_key],
                        expand_k,
                        query_key=selected_key,
                        topk_func=_topk_func,
                    )
                    if not new_edges.empty:
                        combined = pd.concat(
                            [st.session_state.expanded_edges, new_edges],
                            ignore_index=True,
                        ).drop_duplicates()
                        st.session_state.expanded_edges = combined
                    if node_key not in st.session_state.expanded_keys and node_key != selected_key:
                        st.session_state.expanded_keys.append(node_key)
                    st.session_state.focus_key = node_key

    extra_edges = (
        None
        if st.session_state.expanded_edges.empty
        else st.session_state.expanded_edges
    )
    net = build_pyvis_graph(
        song_index,
        selected_key,
        neighbors,
        extra_edges=extra_edges,
        similarity_threshold=similarity_threshold if similarity_threshold > 0 else None,
    )
    vis_network(
        nodes=net.nodes,
        edges=net.edges,
        options=options_to_dict(net.options),
        width=parse_px(net.width, 1200),
        height=parse_px(net.height, 600),
        focus_node=st.session_state.focus_key,
        key="connected_graph",
    )
