from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

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


def _mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return ""
    mode = values.mode()
    if not mode.empty:
        return mode.iloc[0]
    return values.iloc[0]


def _ensure_emotion_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing emotion columns: {missing}")


def _build_song_key(df: pd.DataFrame) -> pd.Series:
    if "id" in df.columns:
        return df["id"].astype(str)

    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
    else:
        title = pd.Series([""] * len(df), index=df.index)

    if "artist" in df.columns:
        artist = df["artist"].fillna("").astype(str)
    else:
        artist = pd.Series([""] * len(df), index=df.index)

    return title + "___" + artist


def build_song_vectors(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    _ensure_emotion_columns(df, EMOTION_COLUMNS)

    df[EMOTION_COLUMNS] = df[EMOTION_COLUMNS].fillna(0.0).astype(float)
    df["song_key"] = _build_song_key(df)

    grouped = df.groupby("song_key", dropna=False)
    vector_means = grouped[EMOTION_COLUMNS].mean()

    meta_cols = [col for col in META_COLUMNS if col in df.columns]
    if meta_cols:
        meta = grouped[meta_cols].agg(_mode_or_first)
    else:
        meta = pd.DataFrame(index=vector_means.index)

    n_lines = grouped.size().rename("n_lines")

    result = pd.concat([meta, vector_means, n_lines], axis=1).reset_index()
    ordered_cols = ["song_key"] + meta_cols + EMOTION_COLUMNS + ["n_lines"]
    result = result[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "data" / "split_song_lyrics_with_BERT2_emotions_en_100k.csv"
    default_output = base_dir / "artifacts" / "song_vectors.parquet"

    parser = argparse.ArgumentParser(description="Build song-level emotion vectors.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    build_song_vectors(args.input, args.output)


if __name__ == "__main__":
    main()
