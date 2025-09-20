from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

import networkx as nx
from .parser_xml import parse_nodes, iter_ways, is_relevant_way, DRIVE_HIGHWAYS, CYCLE_SET
from .utils import compute_haversine_meters, dedupe_consecutive

CAR_TAGS = DRIVE_HIGHWAYS
BIKE_TAGS = CAR_TAGS | CYCLE_SET | {"living_street", "path"}
FOOT_TAGS = BIKE_TAGS | {"pedestrian", "footway", "steps"}

def build_road_network_graph(osm_path: Path) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes: Dict[int, Tuple[float, float]] = parse_nodes(osm_path)

    for osmid, (lat, lon) in nodes.items():
        G.add_node(osmid, y=lat, x=lon)

    for node_ids, tags in iter_ways(osm_path):
        if not is_relevant_way(tags):
            continue

        name: str = tags.get("name", "")
        highway: str = tags.get("highway", "")
        oneway: str = (tags.get("oneway", "no") or "no").lower()

        clean_ids: List[int] = [nid for nid in dedupe_consecutive(node_ids) if nid in nodes]
        if len(clean_ids) < 2:
            continue

        def add_edge(u_node: int, v_node: int):
            lat_a, lon_a = nodes[u_node]
            lat_b, lon_b = nodes[v_node]
            dist_meters: float = compute_haversine_meters(lat_a, lon_a, lat_b, lon_b)
            G.add_edge(u_node, v_node, d=dist_meters, name=name, highway=highway)

        for u, v in zip(clean_ids[:-1], clean_ids[1:]):
            if oneway in {"yes", "true", "1"}:
                add_edge(u, v)
            elif oneway == "-1":
                add_edge(v, u)
            else:
                add_edge(u, v)
                add_edge(v, u)

    return G

class Router:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.nodes_with_coords = {
            node: (data['y'], data['x'])
            for node, data in graph.nodes(data=True)
        }

    def _get_allowed_highways(self, mode: str) -> Set[str]:
        mode = mode.lower()
        if mode == 'car':
            return CAR_TAGS
        if mode == 'bike':
            return BIKE_TAGS
        if mode == 'foot':
            return FOOT_TAGS
        raise ValueError("Modo de transporte inválido. Use 'car', 'bike' ou 'foot'.")

    def find_nearest_node(self, lat: float, lon: float, mode: str) -> Optional[int]:
        allowed_highways = self._get_allowed_highways(mode)
        candidate_nodes: Set[int] = set()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("highway") in allowed_highways:
                candidate_nodes.add(u)
                candidate_nodes.add(v)
        
        if not candidate_nodes:
            return None

        best_id, min_dist = -1, float("inf")
        
        for node_id in candidate_nodes:
            node_lat, node_lon = self.nodes_with_coords[node_id]
            dist = compute_haversine_meters(lat, lon, node_lat, node_lon)
            if dist < min_dist:
                min_dist, best_id = dist, node_id
        
        return best_id

    def find_shortest_path(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        mode: str = 'car'
    ) -> Tuple[Optional[List[int]], float]:
        start_node = self.find_nearest_node(lat1, lon1, mode)
        end_node = self.find_nearest_node(lat2, lon2, mode)

        if start_node is None or end_node is None:
            print("Não foi possível encontrar nós de partida/chegada válidos para o modo.")
            return None, 0.0

        allowed_highways = self._get_allowed_highways(mode)
        
        def edge_filter(u, v):
            edge_data = self.graph.get_edge_data(u, v)
            return edge_data.get('highway') in allowed_highways

        filtered_view = nx.subgraph_view(self.graph, filter_edge=edge_filter)

        try:
            path = nx.dijkstra_path(filtered_view, source=start_node, target=end_node, weight='d')
            path_length = nx.dijkstra_path_length(filtered_view, source=start_node, target=end_node, weight='d')
            return path, path_length
        except nx.NetworkXNoPath:
            print(f"Não há caminho entre {start_node} e {end_node} para o modo '{mode}'.")
            return None, 0.0