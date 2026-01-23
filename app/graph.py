from __future__ import annotations

from typing import Optional

import pandas as pd
from pyvis.network import Network


def _safe_row(song_index: pd.DataFrame, song_key: str) -> Optional[pd.Series]:
    if song_key not in song_index.index:
        return None
    row = song_index.loc[song_key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _make_label(row: Optional[pd.Series], song_key: str) -> tuple[str, str]:
    if row is None:
        return str(song_key), f"song_key: {song_key}"

    title = str(row.get("title", "")) if pd.notna(row.get("title", "")) else ""
    artist = str(row.get("artist", "")) if pd.notna(row.get("artist", "")) else ""
    year = str(row.get("year", "")) if pd.notna(row.get("year", "")) else ""

    # 라벨에 아티스트 포함
    if title and artist:
        label = f"{title}\n{artist}"
    elif title:
        label = title
    else:
        label = str(song_key)
    
    parts = [part for part in [title, artist, year] if part]
    tooltip = " / ".join(parts) if parts else f"song_key: {song_key}"
    if parts:
        tooltip = f"{tooltip}\n(song_key: {song_key})"
    return label, tooltip


def build_pyvis_graph(
    song_index: pd.DataFrame,
    query_key: str,
    neighbors: pd.DataFrame,
    extra_edges: Optional[pd.DataFrame] = None,
    similarity_threshold: Optional[float] = None,
) -> Network:
    net = Network(height="800px", width="100%", bgcolor="#f8f9fa", font_color="#2c3e50")
    net.set_options(
        """
        {
          "nodes": {
            "font": {
              "size": 16,
              "face": "Arial, sans-serif",
              "color": "#2c3e50"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.15)",
              "size": 8,
              "x": 2,
              "y": 2
            },
            "shape": "dot"
          },
          "edges": {
            "color": {
              "color": "rgba(150,150,150,0.4)",
              "highlight": "#6366f1",
              "hover": "#818cf8"
            },
            "width": 1.5,
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.5
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -15000,
              "centralGravity": 0.2,
              "springLength": 180,
              "springConstant": 0.02
            }
          }
        }
        """
    )

    nodes_added = set()

    query_row = _safe_row(song_index, query_key)
    query_label, query_title = _make_label(query_row, query_key)
    net.add_node(
        str(query_key),
        label=query_label,
        title=query_title,
        color={
            "background": "#8b5cf6",
            "border": "#7c3aed",
            "highlight": {
                "background": "#a78bfa",
                "border": "#8b5cf6"
            }
        },
        size=35,
    )
    nodes_added.add(str(query_key))

    for row in neighbors.itertuples(index=False):
        sim = float(row.similarity)
        if similarity_threshold is not None and sim < similarity_threshold:
            continue

        neighbor_key = str(row.song_key)
        neighbor_row = _safe_row(song_index, neighbor_key)
        label, title = _make_label(neighbor_row, neighbor_key)

        if neighbor_key not in nodes_added:
            net.add_node(
                neighbor_key,
                label=label,
                title=title,
                color={
                    "background": "#3b82f6",
                    "border": "#2563eb",
                    "highlight": {
                        "background": "#60a5fa",
                        "border": "#3b82f6"
                    }
                },
                size=12 + sim * 20,
            )
            nodes_added.add(neighbor_key)

        net.add_edge(
            str(query_key),
            neighbor_key,
            value=sim,
            title=f"similarity: {sim:.3f}",
        )

    if extra_edges is not None and not extra_edges.empty:
        for row in extra_edges.itertuples(index=False):
            sim = float(row.similarity)
            if similarity_threshold is not None and sim < similarity_threshold:
                continue

            source_key = str(row.source_key)
            target_key = str(row.target_key)

            for key in [source_key, target_key]:
                if key not in nodes_added:
                    node_row = _safe_row(song_index, key)
                    label, title = _make_label(node_row, key)
                    net.add_node(
                        key,
                        label=label,
                        title=title,
                        color={
                            "background": "#10b981",
                            "border": "#059669",
                            "highlight": {
                                "background": "#34d399",
                                "border": "#10b981"
                            }
                        },
                        size=12,
                    )
                    nodes_added.add(key)

            net.add_edge(
                source_key,
                target_key,
                value=sim,
                title=f"similarity: {sim:.3f}",
            )

    return net
