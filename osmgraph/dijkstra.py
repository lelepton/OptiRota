from _future_ import annotations
from typing import List, Dict, Tuple
import math
import heapq
import networkx as nx

EARTH_RADIUS_M = 6_371_000.0

class NXRoadGraph:
    """Classe para rodar Dijkstra sobre um nx.DiGraph gerado do OSM."""

    def _init_(self, G: nx.DiGraph):
        self.G = G

   
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)*2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)*2
        return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

    # -------------------------------------------------
    # Encontra nó mais próximo de coordenadas
    # -------------------------------------------------
    def nearest_node(self, lat: float, lon: float) -> int:
        best_id = -1
        best_d = float("inf")
        for n, data in self.G.nodes(data=True):
            nlat, nlon = data['y'], data['x']
            d = self.haversine(lat, lon, nlat, nlon)
            if d < best_d:
                best_d, best_id = d, n
        return best_id

    # -------------------------------------------------
    # Dijkstra usando heapq
    # -------------------------------------------------
    def shortest_path_between(self, lat1: float, lon1: float, lat2: float, lon2: float) -> List[Dict]:
        src = self.nearest_node(lat1, lon1)
        dst = self.nearest_node(lat2, lon2)

        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, Tuple[int, Dict]] = {}
        pq: List[Tuple[float, int]] = [(0.0, src)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if u == dst:
                break
            if d_u > dist.get(u, float("inf")):
                continue
            for v in self.G.successors(u):
                data = self.G[u][v]
                nd = d_u + data['d']
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = (u, data)
                    heapq.heappush(pq, (nd, v))

        if dst not in prev and src != dst:
            return []

        # Reconstrói caminho
        path: List[Dict] = []
        cur = dst
        while cur != src:
            if cur not in prev: 
                break
            u, data = prev[cur]
            path.append({
                "u": u,
                "v": cur,
                "d": data['d'],
                "highway": data.get('highway', ''),
                "name": data.get('name', '')
            })
            cur = u

        path.reverse()
        return path