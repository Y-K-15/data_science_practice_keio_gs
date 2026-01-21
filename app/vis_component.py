from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "vis_network"
_vis_network = components.declare_component("vis_network", path=str(_COMPONENT_PATH))


def vis_network(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    options: dict[str, Any],
    width: int,
    height: int,
    focus_node: str | None = None,
    key: str | None = None,
):
    return _vis_network(
        nodes=nodes,
        edges=edges,
        options=options,
        width=width,
        height=height,
        focus_node=focus_node,
        key=key,
    )
