from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from annoy import AnnoyIndex

EMOTION_COLUMNS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

META_COLUMNS = ["title", "artist", "year"]


def _ensure_emotion_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing emotion columns: {missing}")


def _get_meta_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in META_COLUMNS if col in df.columns]
    if "n_lines" in df.columns:
        cols.append("n_lines")
    return cols


def _get_query_index(song_vectors: pd.DataFrame, song_key: str) -> int:
    matches = song_vectors.index[song_vectors["song_key"] == song_key].tolist()
    if not matches:
        raise ValueError(f"song_key not found: {song_key}")
    return matches[0]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms


def angular_distance_to_cosine(distance: float) -> float:
    similarity = 1.0 - (distance * distance) / 2.0
    if similarity > 1.0:
        return 1.0
    if similarity < -1.0:
        return -1.0
    return similarity


def get_topk_similar(
    song_vectors: pd.DataFrame,
    song_key: str,
    k: int = 10,
    similarity_cache: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be >= 1")

    if similarity_cache is not None:
        return get_topk_from_cache(song_vectors, song_key, k, similarity_cache)

    if "song_key" not in song_vectors.columns:
        raise ValueError("song_vectors must include 'song_key'")

    _ensure_emotion_columns(song_vectors, EMOTION_COLUMNS)

    matrix = song_vectors[EMOTION_COLUMNS].to_numpy(dtype=float)
    query_idx = _get_query_index(song_vectors, song_key)

    mat_norm = _normalize_matrix(matrix)
    sims = mat_norm[[query_idx]] @ mat_norm.T
    sims = sims.flatten()
    sims[query_idx] = -np.inf

    max_k = min(k, len(sims) - 1) if len(sims) > 1 else 0
    if max_k <= 0:
        return pd.DataFrame(columns=["similarity", "song_key"] + _get_meta_columns(song_vectors))

    if max_k < len(sims):
        top_idx = np.argpartition(sims, -max_k)[-max_k:]
    else:
        top_idx = np.arange(len(sims))
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    meta_cols = _get_meta_columns(song_vectors)
    result = song_vectors.loc[top_idx, ["song_key"] + meta_cols].copy()
    result.insert(0, "similarity", sims[top_idx])
    result = result.sort_values("similarity", ascending=False).reset_index(drop=True)
    return result


def get_topk_from_cache(
    song_vectors: pd.DataFrame,
    song_key: str,
    k: int,
    similarity_cache: pd.DataFrame,
) -> pd.DataFrame:
    if "song_key" not in similarity_cache.columns or "neighbor_key" not in similarity_cache.columns:
        raise ValueError("similarity_cache must include 'song_key' and 'neighbor_key'")

    subset = similarity_cache[similarity_cache["song_key"] == song_key].copy()
    subset = subset.sort_values("similarity", ascending=False).head(k)
    if subset.empty:
        return pd.DataFrame(columns=["similarity", "song_key"] + _get_meta_columns(song_vectors))

    subset = subset.rename(columns={"song_key": "query_key", "neighbor_key": "song_key"})
    joined = subset.join(song_vectors.set_index("song_key"), on="song_key", how="left")
    joined = joined.drop(columns=["query_key"], errors="ignore")

    meta_cols = _get_meta_columns(song_vectors)
    result = joined[["similarity", "song_key"] + meta_cols].copy()
    return result.reset_index(drop=True)


def build_similarity_cache(
    song_vectors: pd.DataFrame,
    k: int,
    output_path: Path,
    max_songs: int = 5000,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be >= 1")

    if "song_key" not in song_vectors.columns:
        raise ValueError("song_vectors must include 'song_key'")

    _ensure_emotion_columns(song_vectors, EMOTION_COLUMNS)

    n_songs = len(song_vectors)
    if n_songs > max_songs:
        raise ValueError(
            f"Too many songs ({n_songs}) for full cache; increase max_songs if needed."
        )

    matrix = song_vectors[EMOTION_COLUMNS].to_numpy(dtype=float)
    mat_norm = _normalize_matrix(matrix)
    sims = mat_norm @ mat_norm.T
    np.fill_diagonal(sims, -np.inf)

    rows = []
    song_keys = song_vectors["song_key"].astype(str).tolist()

    for i, key in enumerate(song_keys):
        top_idx = np.argsort(sims[i])[::-1][: min(k, n_songs - 1)]
        for j in top_idx:
            rows.append(
                {
                    "song_key": key,
                    "neighbor_key": song_keys[j],
                    "similarity": float(sims[i, j]),
                }
            )

    cache_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(output_path, index=False)
    return cache_df


def build_annoy_index(
    song_vectors: pd.DataFrame,
    n_trees: int = 10,
    metric: str = "angular",
) -> tuple[AnnoyIndex, dict[str, int]]:
    if "song_key" not in song_vectors.columns:
        raise ValueError("song_vectors must include 'song_key'")

    _ensure_emotion_columns(song_vectors, EMOTION_COLUMNS)

    matrix = song_vectors[EMOTION_COLUMNS].to_numpy(dtype=float)
    dims = matrix.shape[1]
    index = AnnoyIndex(dims, metric)
    for idx, vector in enumerate(matrix):
        index.add_item(idx, vector)
    index.build(n_trees)

    keys = song_vectors["song_key"].astype(str).tolist()
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    return index, key_to_index


def get_topk_similar_annoy(
    song_vectors: pd.DataFrame,
    song_key: str,
    k: int,
    annoy_index: AnnoyIndex,
    key_to_index: dict[str, int],
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be >= 1")

    if "song_key" not in song_vectors.columns:
        raise ValueError("song_vectors must include 'song_key'")

    query_idx = key_to_index.get(str(song_key))
    if query_idx is None:
        raise ValueError(f"song_key not found: {song_key}")

    max_items = min(k + 1, len(key_to_index))
    if max_items <= 1:
        return pd.DataFrame(columns=["similarity", "song_key"] + _get_meta_columns(song_vectors))

    indices, distances = annoy_index.get_nns_by_item(
        query_idx, max_items, include_distances=True
    )

    rows = []
    for idx, dist in zip(indices, distances):
        if idx == query_idx:
            continue
        rows.append((idx, angular_distance_to_cosine(float(dist))))
        if len(rows) >= k:
            break

    if not rows:
        return pd.DataFrame(columns=["similarity", "song_key"] + _get_meta_columns(song_vectors))

    row_indices = [idx for idx, _ in rows]
    similarities = [sim for _, sim in rows]
    meta_cols = _get_meta_columns(song_vectors)
    result = song_vectors.iloc[row_indices, :][["song_key"] + meta_cols].copy()
    result.insert(0, "similarity", similarities)
    return result.reset_index(drop=True)
