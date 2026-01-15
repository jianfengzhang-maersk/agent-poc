"""Ontology visualization helpers using NetworkX + PyVis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
from pyvis.network import Network

from agent_poc.semantic_layer.engine import ONTOLOGY_SOURCE_PATH
from agent_poc.semantic_layer.ontology import (
    EntitySchema,
    RelationSchema,
    RelationKey,
    load_ontology,
)

GraphBuild = Tuple[
    nx.DiGraph, Dict[str, EntitySchema], Dict[RelationKey, RelationSchema]
]


def build_nx_graph(ontology_path: str | Path) -> GraphBuild:
    """Return a NetworkX DiGraph populated with ontology metadata."""

    entities, relations = load_ontology(ontology_path)
    graph = nx.DiGraph()

    for idx, (name, schema) in enumerate(entities.items()):
        graph.add_node(
            name,
            label=name,
            title=(schema.description or name),
            size=25,
            group=idx % 4,
            attributes=", ".join(schema.attributes) if schema.attributes else "",
        )

    for (source, rel_name, target), schema in relations.items():
        graph.add_edge(
            source,
            target,
            label=rel_name,
            title=schema.description or rel_name,
            weight=1.0,
        )

    return graph, entities, relations


def render_pyvis_network(
    graph: nx.DiGraph,
    html_path: str | Path = "ontology_network.html",
    height: str = "700px",
    width: str = "100%",
    physics: bool = True,
    show_buttons: bool = True,
    node_size: int = 12,
    node_font_size: int = 14,
    edge_width: float = 2.0,
    edge_font_size: int = 12,
) -> Network:
    """Convert the NetworkX graph into a PyVis interactive network."""

    net = Network(height=height, width=width, notebook=False, directed=True)
    net.from_nx(graph)

    for node in net.nodes:
        node["size"] = node_size
        node.setdefault("font", {})
        node["font"].update({"size": node_font_size})

    for edge in net.edges:
        edge["width"] = edge_width
        edge.setdefault("font", {})
        edge["font"].update({"size": edge_font_size, "align": "horizontal"})

    if show_buttons:
        net.show_buttons(filter_=["physics"])
    net.toggle_physics(physics)

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(html_path))
    return net


if __name__ == "__main__":
    ontology_path = ONTOLOGY_SOURCE_PATH
    nx_graph, _, _ = build_nx_graph(ontology_path)

    render_pyvis_network(
        nx_graph,
        html_path="src/agent_poc/semantic_layer/ontology.html",
        height="750px",
        width="100%",
        physics=True,
        node_size=10,
        node_font_size=8,
        edge_width=1.5,
        edge_font_size=6,
    )
