from pathlib import Path
from typing import Dict, Tuple, List
import networkx as nx

from .parser_xml import parse_nodes, iter_ways, is_relevant_way
from .utils import compute_haversine_meters, dedupe_consecutive

def build_graph_in_memory(osm_path: Path) -> nx.DiGraph:
    """
    Cria um grafo dirigido em memória a partir de um arquivo OSM.
    Nodes: osmid, latitude, longitude
    Edges: u, v, d (distância), name, highway
    """
    G: nx.DiGraph = nx.DiGraph()
    nodes: Dict[int, Tuple[float, float]] = parse_nodes(osm_path)

    
    for osmid, (lat, lon) in nodes.items():
        G.add_node(osmid, y=lat, x=lon)

  
    for node_ids, tags in iter_ways(osm_path):
        if not is_relevant_way(tags):
            continue

        name: str = tags.get("name", "") or ""
        highway: str = tags.get("highway", "") or ""
        oneway: str = (tags.get("oneway", "no") or "").lower()

        clean_ids: List[int] = [nid for nid in dedupe_consecutive(node_ids) if nid in nodes]
        if len(clean_ids) < 2:
            continue

        def add_edge(a: int, b: int) -> None:
            lat_a, lon_a = nodes[a]
            lat_b, lon_b = nodes[b]
            d_m: float = compute_haversine_meters(lat_a, lon_a, lat_b, lon_b)
            G.add_edge(a, b, d=d_m, name=name, highway=highway)

        for u, v in zip(clean_ids[:-1], clean_ids[1:]):
            if oneway in {"yes", "true", "1"}:
                add_edge(u, v)
            elif oneway == "-1":
                add_edge(v, u)
            else:
                add_edge(u, v)
                add_edge(v, u)

    return G