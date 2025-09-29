from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import time
import pandas as pd
import math
import os
import sys

# -------------------------------------------------------------------------
# Constantes: tipos de via por highway
# -------------------------------------------------------------------------
CAR_TAGS = {
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "service",
}
BIKE_TAGS = CAR_TAGS | {"living_street", "cycleway", "path"}
# Observação: pedestre (foot) inclui todas as vias, mantemos allow_foot=True.

# ------------------------------
# Velocidades por tipo de via (m/s) usadas na estimativa de tempo por aresta
SPEED_LIMITS_MPS = {
    "primary":     22.22,  # ~80 km/h
    "secondary":   16.67,  # ~60 km/h
    "tertiary":    11.11,  # ~40 km/h
    "residential":  8.33,  # ~30 km/h
}

# ------------------------------
# Coeficientes (heurística de "tempo" artificial) por tipo de via base
COEF_BY_TAG = {
    "primary": 0.8,
    "secondary": 0.6,
    "tertiary": 0.4,
    "residential": 0.3,
}

# -------------------------------------------------------------------------
# Funções utilitárias (reutilizáveis)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Função: _normalize_tag
# -------------------------------------------------------------------------
# Descrição
# ---------
# Normaliza a tag de highway para um dos valores-base esperados
# ("primary", "secondary", "tertiary", "residential"). Qualquer outra
# tag cai em "residential" por padrão.
#
# Parâmetros
# ----------
# tag : str (tag original do CSV)
#
# Retorno
# -------
# str : tag normalizada
def _normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    return t if t in {"primary", "secondary", "tertiary", "residential"} else "residential"


# -------------------------------------------------------------------------
# Função: _speed_mps_for_tag
# -------------------------------------------------------------------------
# Descrição
# ---------
# Retorna a velocidade (m/s) associada à tag normalizada do tipo de via.
#
# Parâmetros
# ----------
# tag : str (tag original do CSV)
#
# Retorno
# -------
# float : velocidade em m/s
def _speed_mps_for_tag(tag: str) -> float:
    return SPEED_LIMITS_MPS.get(_normalize_tag(tag))


# -------------------------------------------------------------------------
# Função: _coef_for_tag
# -------------------------------------------------------------------------
# Descrição
# ---------
# Retorna o coeficiente usado no custo artificial da A* com tempo,
# associado à tag normalizada do tipo de via.
#
# Parâmetros
# ----------
# tag : str (tag original do CSV)
#
# Retorno
# -------
# float : coeficiente (adimensional)
def _coef_for_tag(tag: str) -> float:
    return COEF_BY_TAG.get(_normalize_tag(tag))


# -------------------------------------------------------------------------
# Função: _haversine_m
# -------------------------------------------------------------------------
# Descrição
# ---------
# Calcula a distância Haversine entre dois pontos (lat, lon) em metros.
#
# Parâmetros
# ----------
# lat1, lon1 : float (ponto A)
# lat2, lon2 : float (ponto B)
#
# Retorno
# -------
# float : distância em metros
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * 6371008.8 * math.asin(math.sqrt(a))


# -------------------------------------------------------------------------
# Função: _is_edge_allowed
# -------------------------------------------------------------------------
# Descrição
# ---------
# Informa se uma aresta é permitida para o modo de viagem informado.
#
# Parâmetros
# ----------
# travel_mode : str ('car' | 'bike' | 'foot')
# edge        : Edge
#
# Retorno
# -------
# bool : True se a aresta pode ser usada no modo; False caso contrário
def _is_edge_allowed(travel_mode: str, edge: "Edge") -> bool:
    mode = travel_mode.lower()
    if mode == "car":
        return edge.allow_car
    if mode == "bike":
        return edge.allow_bike or edge.allow_car
    # foot
    return edge.allow_foot or edge.allow_bike or edge.allow_car


# -------------------------------------------------------------------------
# Estrutura: Edge
# -------------------------------------------------------------------------
# Descrição
# ---------
# Aresta dirigida u->v com:
# - v (destino), w (peso 'd' em metros), tag (highway)
# - flags de acesso: allow_car, allow_bike, allow_foot
# - name: nome da via (pode ser "")
#
# Observações
# -----------
# Dataclass imutável (frozen=True) para facilitar depuração e segurança.
@dataclass(frozen=True)
class Edge:
    v: int
    w: float
    tag: str
    allow_car: bool
    allow_bike: bool
    allow_foot: bool
    name: str


# -------------------------------------------------------------------------
# Classe: RoadGraph
# -------------------------------------------------------------------------
# Descrição
# ---------
# Representa o grafo dirigido e ponderado G=(V,E) carregado a partir de CSVs.
# - nodes[osmid] = (lat, lon)
# - adj[u] = lista de objetos Edge saindo de u
# - w = coluna 'd' (metros); não recalculamos distância neste módulo.
#
# Observações
# -----------
# Cada aresta possui um único 'highway' e suas flags de acesso são definidas
# a partir dos conjuntos CAR_TAGS/BIKE_TAGS; allow_foot é sempre True.
class RoadGraph:
    # -------------------------------------------------------------------------
    # Função: __init__
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Inicializa contêineres de nós e adjacência.
    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.adj: Dict[int, List[Edge]] = {}

    # -------------------------------------------------------------------------
    # Função: coef_for
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Obtém o coeficiente associado a uma tag de via.
    #
    # Parâmetros
    # ----------
    # tag : str (highway)
    #
    # Retorno
    # -------
    # float : coeficiente usado no peso artificial (A* com tempo)
    def coef_for(self, tag: str) -> float:
        return _coef_for_tag(tag)

    # -------------------------------------------------------------------------
    # Função utilitária (reutilizável): heuristic_distance_to_goal
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Retorna a distância Haversine (em metros) entre um nó "node_id" e o nó
    # "target_node_id". Usada como heurística admissível para A*.
    def heuristic_distance_to_goal(self, node_id: int, target_node_id: int) -> float:
        node_lat, node_lon = self.nodes[node_id]
        target_lat, target_lon = self.nodes[target_node_id]
        return _haversine_m(node_lat, node_lon, target_lat, target_lon)
    
    # -------------------------------------------------------------------------
    # Função: distance_from_node_to_point_m
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Calcula a distância Haversine (em metros) entre um nó existente do grafo
    # (identificado por 'node_id') e um ponto arbitrário (lat, lon).
    #
    # Parâmetros
    # ----------
    # node_id : int   (identificador do nó no grafo)
    # lat     : float (latitude do ponto-alvo, em graus decimais)
    # lon     : float (longitude do ponto-alvo, em graus decimais)
    #
    # Retorno
    # -------
    # float | None : distância em metros; None se o nó não existir no grafo
    def distance_from_node_to_point_m(self, node_id: int, lat: float, lon: float) -> float | None:
        if node_id not in self.nodes:
            return None
        node_lat, node_lon = self.nodes[node_id]
        return _haversine_m(node_lat, node_lon, float(lat), float(lon))


    # -------------------------------------------------------------------------
    # Função: load_graph
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Lê os CSVs de nós (osmid,y,x) e arestas (u,v,d,highway[,name]) e constrói o grafo.
    #
    # Parâmetros
    # ----------
    # nodes_csv : caminho do CSV de nós
    # edges_csv : caminho do CSV de arestas
    #
    # Retorno
    # -------
    # RoadGraph : instância pronta para consulta
    #
    # Observações
    # -----------
    # - Não recalcula distância; usa 'd' como peso.
    @classmethod
    def load_graph(cls, nodes_csv: str, edges_csv: str) -> "RoadGraph":
        graph = cls()

        # Nós
        nodes_df = pd.read_csv(nodes_csv)
        graph.nodes = {
            int(osmid): (float(lat), float(lon))
            for osmid, lat, lon in zip(nodes_df["osmid"], nodes_df["y"], nodes_df["x"])
        }

        # Arestas
        edges_df = pd.read_csv(edges_csv)
        has_name = "name" in edges_df.columns

        for _, row in edges_df.iterrows():
            source_id, dest_id, weight_m = int(row["u"]), int(row["v"]), float(row["d"])
            tag = str(row["highway"]).strip()
            name = "" if (not has_name or pd.isna(row.get("name", ""))) else str(row["name"])

            edge = Edge(
                v=dest_id, w=weight_m, tag=tag,
                allow_car=(tag in CAR_TAGS),
                allow_bike=(tag in BIKE_TAGS),
                allow_foot=True,
                name=name,
            )
            graph.adj.setdefault(source_id, []).append(edge)

        return graph

    # -------------------------------------------------------------------------
    # Função: neighbors
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Retorna as arestas saindo de um nó u, opcionalmente filtradas por 'highway'.
    #
    # Parâmetros
    # ----------
    # u       : id do nó de origem
    # highway : 'car' | 'bike' | 'foot' | None
    #
    # Retorno
    # -------
    # Iterable[Edge] : lista de arestas (filtrada quando 'highway' é fornecido)
    def neighbors(self, u: int, highway: Optional[str] = None) -> Iterable[Edge]:
        outgoing_edges = self.adj.get(u, [])
        if highway is None:
            return list(outgoing_edges)
        mode = highway.lower()
        if mode == "car":
            return [edge for edge in outgoing_edges if edge.allow_car]
        if mode == "bike":
            return [edge for edge in outgoing_edges if edge.allow_bike or edge.allow_car]
        return [edge for edge in outgoing_edges if edge.allow_foot or edge.allow_bike or edge.allow_car]  # foot

    # -------------------------------------------------------------------------
    # Função: edge_count
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Calcula a quantidade total de arestas do grafo.
    #
    # Retorno
    # -------
    # int : |E| (soma dos comprimentos das listas de adjacência)
    def edge_count(self) -> int:
        return sum(len(edge_list) for edge_list in self.adj.values())

    # -------------------------------------------------------------------------
    # Função: node_count
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Calcula a quantidade total de nós do grafo.
    #
    # Retorno
    # -------
    # int : |V| (tamanho de nodes)
    def node_count(self) -> int:
        return len(self.nodes)

    # -------------------------------------------------------------------------
    # Função: nearest_node
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Encontra o osmid do nó mais próximo às coordenadas (lat, lon), respeitando
    # o filtro 'highway' (car | bike | foot).
    #
    # Parâmetros
    # ----------
    # lat     : latitude em graus decimais
    # lon     : longitude em graus decimais
    # highway : 'car' | 'bike' | 'foot'
    #
    # Retorno
    # -------
    # int : osmid do nó mais próximo entre os candidatos
    #
    # Observações
    # -----------
    # Inclusão por highway:
    # - car  → allow_car
    # - bike → allow_bike OU allow_car
    # - foot → allow_foot OU allow_bike OU allow_car (todas)
    def nearest_node(self, lat: float, lon: float, highway: str) -> int:
        travel_mode = str(highway).strip().lower()

        # Candidatos: endpoints das arestas válidas para o filtro
        candidate_node_ids: set[int] = set()
        for source_id, edges in self.adj.items():
            for edge in edges:
                if _is_edge_allowed(travel_mode, edge):
                    candidate_node_ids.update([source_id, edge.v])

        # Busca linear por Haversine
        best_node_id, best_distance_m = -1, float("inf")
        for node_id in candidate_node_ids:
            node_lat, node_lon = self.nodes[node_id]
            distance_m = _haversine_m(float(lat), float(lon), node_lat, node_lon)
            if distance_m < best_distance_m:
                best_distance_m, best_node_id = distance_m, node_id
        return best_node_id

    # -------------------------------------------------------------------------
    # Função: _reconstruct_path (interna)
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Reconstrói a lista de arestas do caminho a partir de um dicionário de
    # predecessores {nó: (nó_anterior, aresta_usada)}.
    def _reconstruct_path(self, predecessor: dict[int, tuple[int, Edge]],
                          source_node_id: int, target_node_id: int) -> list[Edge]:
        path_edges: list[Edge] = []
        node_cursor = target_node_id
        while node_cursor != source_node_id:
            if node_cursor not in predecessor:
                break
            prev_node_id, edge_used = predecessor[node_cursor]
            path_edges.append(edge_used)
            node_cursor = prev_node_id
        path_edges.reverse()
        return path_edges

    # -------------------------------------------------------------------------
    # Função: shortest_path_between
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Executa Dijkstra entre dois pares (lat, lon), restringindo às arestas
    # válidas para o 'highway' informado.
    #
    # Parâmetros
    # ----------
    # lat1, lon1 : coordenadas da origem
    # lat2, lon2 : coordenadas do destino
    # highway    : 'car' | 'bike' | 'foot'
    #
    # Retorno
    # -------
    # list[Edge] : lista de arestas na ordem origem→destino (vazia se não houver caminho)
    #
    # Observações
    # -----------
    # Passos: mapear coordenadas p/ nós com nearest_node; rodar Dijkstra; reconstruir a rota.
    def shortest_path_between(self, lat1: float, lon1: float,
                              lat2: float, lon2: float,
                              highway: str) -> list[Edge]:
        import heapq
        source_node_id = self.nearest_node(lat1, lon1, highway)
        target_node_id = self.nearest_node(lat2, lon2, highway)

        INF = float("inf")
        travel_mode = str(highway).strip().lower()

        # Estruturas de Dijkstra
        distance_from_source: dict[int, float] = {source_node_id: 0.0}
        predecessor: dict[int, tuple[int, Edge]] = {}
        priority_queue: list[tuple[float, int]] = [(0.0, source_node_id)]

        while priority_queue:
            current_distance, current_node_id = heapq.heappop(priority_queue)
            if current_node_id == target_node_id:
                break
            if current_distance > distance_from_source.get(current_node_id, INF):
                continue

            for edge in self.adj.get(current_node_id, []):
                if not _is_edge_allowed(travel_mode, edge):
                    continue

                neighbor_node_id = edge.v
                new_distance = current_distance + edge.w
                if new_distance < distance_from_source.get(neighbor_node_id, INF):
                    distance_from_source[neighbor_node_id] = new_distance
                    predecessor[neighbor_node_id] = (current_node_id, edge)
                    heapq.heappush(priority_queue, (new_distance, neighbor_node_id))

        if target_node_id not in predecessor and source_node_id != target_node_id:
            return []

        return self._reconstruct_path(predecessor, source_node_id, target_node_id)

    # -------------------------------------------------------------------------
    # Função: astar_shortest_path_between
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Executa A* entre dois pares (lat, lon) usando a heurística de Haversine
    # (distância em linha reta) até o destino, respeitando o filtro 'highway'.
    #
    # Parâmetros
    # ----------
    # lat1, lon1 : coordenadas da origem
    # lat2, lon2 : coordenadas do destino
    # highway    : 'car' | 'bike' | 'foot'
    #
    # Retorno
    # -------
    # list[Edge] : lista de arestas na ordem origem→destino (vazia se não houver caminho)
    #
    # Observações
    # -----------
    # - Heurística Haversine é admissível/consistente quando o custo é distância.
    # - f(n) = g(n) + h(n), onde g(n) é a soma de pesos e h(n) é Haversine(n, destino).
    def astar_shortest_path_between(self, lat1: float, lon1: float,
                                    lat2: float, lon2: float,
                                    highway: str) -> list[Edge]:
        import heapq

        # Mapear coordenadas para nós
        source_node_id = self.nearest_node(lat1, lon1, highway)
        target_node_id = self.nearest_node(lat2, lon2, highway)

        # Heurística: reutiliza método da classe (Haversine)
        λ = lambda n: self.heuristic_distance_to_goal(n, target_node_id)

        INF = float("inf")
        travel_mode = str(highway).strip().lower()

        # Estruturas do A*
        distance_from_source: dict[int, float] = {source_node_id: 0.0}
        predecessor: dict[int, tuple[int, Edge]] = {}
        # heap com (f=g+λ, g, node)
        initial_heuristic_distance = λ(source_node_id)
        frontier_heap: list[tuple[float, float, int]] = [(initial_heuristic_distance, 0.0, source_node_id)]
        expanded_nodes = set()

        while frontier_heap:
            _, cost_from_start, current_node_id = heapq.heappop(frontier_heap)
            if current_node_id in expanded_nodes:
                continue
            expanded_nodes.add(current_node_id)

            if current_node_id == target_node_id:
                break

            # Expansão com filtro por highway
            for edge in self.adj.get(current_node_id, []):
                if not _is_edge_allowed(travel_mode, edge):
                    continue

                neighbor_node_id = edge.v
                tentative_cost_from_start = cost_from_start + edge.w
                if tentative_cost_from_start < distance_from_source.get(neighbor_node_id, float("inf")):
                    distance_from_source[neighbor_node_id] = tentative_cost_from_start
                    predecessor[neighbor_node_id] = (current_node_id, edge)
                    estimated_total_cost_neighbor = tentative_cost_from_start + λ(neighbor_node_id)
                    heapq.heappush(frontier_heap, (estimated_total_cost_neighbor, tentative_cost_from_start, neighbor_node_id))

        if target_node_id not in predecessor and source_node_id != target_node_id:
            return []

        return self._reconstruct_path(predecessor, source_node_id, target_node_id)

    # -------------------------------------------------------------------------
    # Função: astar_with_time_between
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Executa A* com custo artificial (baseado em coeficientes por tag) e
    # acumula o tempo real percorrido simultaneamente.
    #
    # Parâmetros
    # ----------
    # lat1, lon1 : coordenadas da origem
    # lat2, lon2 : coordenadas do destino
    # highway    : 'car' | 'bike' | 'foot'
    #
    # Retorno
    # -------
    # tuple[list[Edge], float, float] : (arestas, distância_total_m, tempo_total_s)
    #
    # Observações
    # -----------
    # - O "peso" minimizado é g = Σ (w / coef(tag)), enquanto o tempo real é somado à parte.
    # - A heurística λ(n) = Haversine(n, destino) permanece sem escala/divisão.
    def astar_with_time_between(self, lat1: float, lon1: float,
                                lat2: float, lon2: float,
                                highway: str) -> tuple[list[Edge], float, float]:
        import heapq

        # Mapear coordenadas para nós
        source_node_id = self.nearest_node(lat1, lon1, highway)
        target_node_id = self.nearest_node(lat2, lon2, highway)

        # Heurística: reutiliza método da classe (Haversine) — sem dividir por constantes
        λ = lambda n: self.heuristic_distance_to_goal(n, target_node_id)

        travel_mode = str(highway).strip().lower()

        # Estruturas do A*: g = peso artificial acumulado; heap carrega tempo acumulado em paralelo
        distance_from_source: dict[int, float] = {source_node_id: 0.0}
        predecessor: dict[int, tuple[int, Edge]] = {}
        time_from_source: dict[int, float] = {source_node_id: 0.0}

        # heap com (f=g+λ, g, node, accumulated_time_seconds)
        initial_f = λ(source_node_id)
        frontier_heap: list[tuple[float, float, int, float]] = [(initial_f, 0.0, source_node_id, 0.0)]
        expanded_nodes = set()

        while frontier_heap:
            _, cost_from_start, current_node_id, accumulated_time_seconds = heapq.heappop(frontier_heap)
            if current_node_id in expanded_nodes:
                continue
            expanded_nodes.add(current_node_id)

            if current_node_id == target_node_id:
                break

            # Expansão com filtro por highway
            for edge in self.adj.get(current_node_id, []):
                if not _is_edge_allowed(travel_mode, edge):
                    continue

                # custo artificial da aresta e tempo (real) em paralelo
                edge_weight = edge.w / self.coef_for(edge.tag)                 # critério minimizado (peso artificial)
                edge_time_seconds = edge.w / _speed_mps_for_tag(edge.tag)      # tempo real (segundos)

                neighbor_node_id = edge.v
                tentative_cost_from_start = cost_from_start + edge_weight
                if tentative_cost_from_start < distance_from_source.get(neighbor_node_id, float("inf")):
                    distance_from_source[neighbor_node_id] = tentative_cost_from_start
                    predecessor[neighbor_node_id] = (current_node_id, edge)
                    time_from_source[neighbor_node_id] = accumulated_time_seconds + edge_time_seconds
                    estimated_total_cost_neighbor = tentative_cost_from_start + λ(neighbor_node_id)
                    heapq.heappush(
                        frontier_heap,
                        (estimated_total_cost_neighbor,
                         tentative_cost_from_start,
                         neighbor_node_id,
                         accumulated_time_seconds + edge_time_seconds)
                    )

        if target_node_id not in predecessor and source_node_id != target_node_id:
            return [], 0.0, float("inf")

        # Reconstrução do caminho + distância/tempo
        path_edges = self._reconstruct_path(predecessor, source_node_id, target_node_id)
        total_distance_m = sum(e.w for e in path_edges)
        total_time_s = time_from_source.get(target_node_id, 0.0)
        return path_edges, total_distance_m, total_time_s


# -------------------------------------------------------------------------
# Funções auxiliares de impressão e formatação
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Função: _format_seconds_hms
# -------------------------------------------------------------------------
# Descrição
# ---------
# Formata segundos em 'Hh MMmin SSs', 'Mmin SSs' ou 'Ss' conforme o caso.
#
# Parâmetros
# ----------
# total_seconds : float
#
# Retorno
# -------
# str : string amigável de duração
def _format_seconds_hms(total_seconds: float) -> str:
    seconds = int(round(total_seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}min {secs:02d}s"
    if minutes > 0:
        return f"{minutes}min {secs:02d}s"
    return f"{secs}s"


# -------------------------------------------------------------------------
# Função: print_path_edges
# -------------------------------------------------------------------------
# Descrição
# ---------
# Imprime a sequência de arestas com distância por aresta, distância
# cumulativa até então e metadados, além da distância e tempo totais ao final.
#
# Parâmetros
# ----------
# path_edges      : list[Edge] (na ordem origem→destino)
# source_node_id  : int (id do nó de origem para imprimir "u -> v")
# label_total     : str (rótulo do total, ex.: "Dijkstra" ou "A*")
def print_path_edges(path_edges, source_node_id: int, label_total: str) -> None:
    cumulative_distance = 0.0
    cumulative_time_s = 0.0
    running_source = source_node_id
    for edge in path_edges:
        cumulative_distance += edge.w
        edge_time_s = edge.w / _speed_mps_for_tag(edge.tag)
        cumulative_time_s += edge_time_s
        edge_time_fmt = _format_seconds_hms(edge_time_s)
        print(f"{running_source} -> {edge.v} | w={edge.w:.3f} m | t≈{edge_time_fmt} | cumulativo={cumulative_distance:.3f} m | tag={edge.tag} | name=\"{edge.name}\"")
        running_source = edge.v
    print(f"Distância total ({label_total}): {cumulative_distance:.3f} m")
    print(f"Tempo total estimado ({label_total}): {_format_seconds_hms(cumulative_time_s)} ({cumulative_time_s:.1f} s)")


# -------------------------------------------------------------------------
# Estrutura: Delivery (janelas diárias + peso)
# -------------------------------------------------------------------------
@dataclass
class Delivery:
    delivery_identifier: int
    delivery_latitude: float
    delivery_longitude: float
    delivery_weight_tons: float
    delivery_time_windows_seconds: list[tuple[int, int]]
    delivery_was_completed: bool = False

    # -------------------------------------------------------------------------
    # Função (método estático): Delivery._time_to_seconds
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Converte uma string de horário "HH:MM" para o número de segundos desde 00:00.
    #
    # Parâmetros
    # ----------
    # time_hhmm : str
    #
    # Retorno
    # -------
    # int : segundos no dia (0–86399)
    @staticmethod
    def _time_to_seconds(time_hhmm: str) -> int:
        hh, mm = [int(x) for x in time_hhmm.strip().split(":")]
        return hh * 3600 + mm * 60

    # -------------------------------------------------------------------------
    # Função (método de classe): Delivery.from_windows_string
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Constrói um objeto Delivery a partir de uma string de janelas no formato:
    # "HH:MM-HH:MM, HH:MM-HH:MM, ..." e de um peso em toneladas.
    # As janelas são convertidas para segundos e janelas sobrepostas são mescladas.
    #
    # Parâmetros
    # ----------
    # idx            : int             (identificador)
    # lat, lon       : float           (coordenadas do cliente)
    # windows_string : str             (janelas em string)
    # weight_tons    : float           (peso da entrega)
    #
    # Retorno
    # -------
    # Delivery : instância preenchida
    @classmethod
    def from_windows_string(cls, idx: int, lat: float, lon: float,
                            windows_string: str, weight_tons: float) -> "Delivery":
        pairs: list[tuple[int, int]] = []
        for seg in [s.strip() for s in windows_string.split(",") if s.strip()]:
            a, b = [p.strip() for p in seg.split("-")]
            s, e = cls._time_to_seconds(a), cls._time_to_seconds(b)
            if e < s:
                s, e = e, s
            pairs.append((s, e))
        pairs.sort()
        # mesclar sobreposições
        merged: list[tuple[int, int]] = []
        for s, e in pairs:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        return cls(idx, float(lat), float(lon), float(weight_tons), merged)

    # -------------------------------------------------------------------------
    # Função (método): Delivery.earliest_same_day_service_time
    # -------------------------------------------------------------------------
    # Descrição
    # ---------
    # Dado um horário de chegada (em segundos do dia), retorna o primeiro instante
    # **no mesmo dia** em que a entrega pode ser atendida.
    #
    # Parâmetros
    # ----------
    # arrival_seconds_of_day : float
    #
    # Retorno
    # -------
    # int | None : segundos do dia do atendimento; None se todas as janelas de hoje
    #              já passaram para esse horário de chegada.
    def earliest_same_day_service_time(self, arrival_seconds_of_day: float) -> int | None:
        best: int | None = None
        for s, e in self.delivery_time_windows_seconds:
            if arrival_seconds_of_day > e:
                continue
            candidate = int(arrival_seconds_of_day) if arrival_seconds_of_day >= s else s
            if best is None or candidate < best:
                best = candidate
        return best


# -------------------------------------------------------------------------
# Utilitários de tempo (locais, sem conflito)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Função: _time_to_seconds
# -------------------------------------------------------------------------
# Descrição
# ---------
# Converte "HH:MM" em segundos desde 00:00, para uso geral no módulo.
#
# Parâmetros
# ----------
# time_hhmm : str
#
# Retorno
# -------
# int : segundos no dia (0–86399)
def _time_to_seconds(time_hhmm: str) -> int:
    hh, mm = [int(x) for x in time_hhmm.strip().split(":")]
    return hh * 3600 + mm * 60


# -------------------------------------------------------------------------
# Função: _seconds_to_hhmmss
# -------------------------------------------------------------------------
# Descrição
# ---------
# Converte segundos do dia (float/int) no formato de string "HH:MM:SS".
#
# Parâmetros
# ----------
# secs : float
#
# Retorno
# -------
# str : horário formatado "HH:MM:SS"
def _seconds_to_hhmmss(secs: float) -> str:
    s = int(secs)
    hh = (s // 3600) % 24
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


# -------------------------------------------------------------------------
# Função: _split_minutes_seconds_exact
# -------------------------------------------------------------------------
# Descrição
# ---------
# Separa uma duração em segundos em (minutos inteiros, segundos remanescentes),
# sem arredondamentos externos, preservando exatidão.
#
# Parâmetros
# ----------
# total_seconds : float
#
# Retorno
# -------
# tuple[int, float] : (minutos_inteiros, segundos_restantes)
def _split_minutes_seconds_exact(total_seconds: float) -> tuple[int, float]:
    mins = int(total_seconds // 60)
    rem = total_seconds - mins * 60
    return mins, rem


# -------------------------------------------------------------------------
# Função: _fmt_seconds
# -------------------------------------------------------------------------
# Descrição
# ---------
# Formata a parte de segundos remanescentes para impressão amigável:
# sem casas se inteiro; até 3 casas sem zeros à direita caso contrário.
#
# Parâmetros
# ----------
# rem : float
#
# Retorno
# -------
# str : representação amigável dos segundos
def _fmt_seconds(rem: float) -> str:
    return str(int(round(rem))) if abs(rem - round(rem)) < 1e-9 else f"{rem:.3f}".rstrip("0").rstrip(".")


# -------------------------------------------------------------------------
# Parser do arquivo de entrada
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Função: parse_vrp_input_file
# -------------------------------------------------------------------------
# Descrição
# ---------
# Lê um arquivo texto no formato:
#   <depot_lat> <depot_lon> <capacity_tons>
#
#   <lat> <lon> "HH:MM-HH:MM, HH:MM-HH:MM" <weight_tons>
#   ...
# Retorna tupla com origem, capacidade e lista de entregas cruas.
#
# Parâmetros
# ----------
# file_path : str (caminho do arquivo)
#
# Retorno
# -------
# (float, float, float, list[list]) :
#   (origin_lat, origin_lon, capacity_tons, deliveries_raw)
#   onde deliveries_raw = [[lat, lon, weight_tons, windows_string], ...]
#
# Observações
# -----------
# - Não há validação/erros; assume-se arquivo bem formatado.
def parse_vrp_input_file(file_path: str) -> tuple[float, float, float, list[list]]:
    import re
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    m0 = re.match(r'^([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)$', lines[0])
    origin_lat, origin_lon, capacity_tons = float(m0.group(1)), float(m0.group(2)), float(m0.group(3))

    rx = re.compile(r'^([-\d.]+)\s*,?\s+([-\d.]+)\s+"([^"]+)"\s+([-\d.]+)$')
    deliveries_raw: list[list] = []
    for idx, line in enumerate(lines[1:]):
        m = rx.match(line)
        lat = float(m.group(1)); lon = float(m.group(2))
        windows_string = m.group(3).strip()
        weight_tons = float(m.group(4))
        deliveries_raw.append([lat, lon, weight_tons, windows_string])
    return origin_lat, origin_lon, capacity_tons, deliveries_raw


# -------------------------------------------------------------------------
# Função: VRP_heuristic
# -------------------------------------------------------------------------
# Descrição
# ---------
# Heurística gulosa para roteirização com janelas de tempo diárias e capacidade
# com reset no depósito. O algoritmo simula “teletransporte”: a posição do
# veículo salta diretamente para o ponto de atendimento e o relógio avança
# apenas pelo tempo de viagem calculado.
#
# Regras de escolha
# -----------------
# 1) Entre as entregas atendíveis **agora** (sem espera), escolha a que tem
#    menor horário de atendimento (empate: menor distância).
# 2) Se não houver atendimento imediato, considere **voltar ao depósito**
#    (somando o tempo de retorno) e planejar a partir de lá (aqui pode haver
#    espera até a janela). Escolha a de menor horário de atendimento após o
#    retorno (empate: menor distância).
# 3) Se nada for possível HOJE (mesmo considerando o retorno), o caminhão
#    retorna à base (somando o tempo), e avançamos para o próximo dia (DIA N+1),
#    reiniciando o relógio em start_time_HH_MM e resetando a capacidade.
#
# Observações
# -----------
# - A capacidade NÃO entra no ranking; só impede “atendimento imediato” se não
#   couber. Após retorno ao depósito, a capacidade é resetada.
# - Assume-se que **não existem entregas que comecem em um dia e terminem em outro**.
#
# Parâmetros
# ----------
# graph            : RoadGraph
# origin_lat       : float (latitude do depósito)
# origin_lon       : float (longitude do depósito)
# start_time_HH_MM : str   (horário de partida "HH:MM")
# capacity_tons_L  : float (capacidade máxima do caminhão em toneladas)
# deliveries_raw   : list  ([[lat, lon, weight_tons, windows_string], ...])
# highway          : str   ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None (imprime o plano por "DIA N")
def VRP_heuristic(
    graph: "RoadGraph",
    origin_lat: float,
    origin_lon: float,
    start_time_HH_MM: str,
    capacity_tons_L: float,
    deliveries_raw: list[list],
    highway: str = "car",
) -> None:
    # 1) Construir objetos Delivery a partir das entradas cruas
    deliveries: list[Delivery] = [
        Delivery.from_windows_string(idx, lat, lon, windows_string, weight_tons)
        for idx, (lat, lon, weight_tons, windows_string) in enumerate(deliveries_raw)
    ]

    # 2) Estado da simulação
    current_latitude, current_longitude = float(origin_lat), float(origin_lon)
    current_planning_day = 1
    current_time_seconds_of_day = float(_time_to_seconds(start_time_HH_MM))
    current_vehicle_remaining_capacity = float(capacity_tons_L)

    print(f"DIA {current_planning_day}")

    while True:
        pending_deliveries = [d for d in deliveries if not d.delivery_was_completed]
        if not pending_deliveries:
            # Retorno final ao depósito também consome tempo
            _unused_edges, _unused_distance_meters, travel_seconds_back_to_depot = graph.astar_with_time_between(
                current_latitude, current_longitude, origin_lat, origin_lon, highway
            )
            current_time_seconds_of_day += travel_seconds_back_to_depot
            print("Carga retorna a distribuidora.")
            break

        # Tempo de retorno ao depósito a partir da posição atual
        _unused_e_ret, _unused_d_ret, travel_seconds_back_to_depot = graph.astar_with_time_between(
            current_latitude, current_longitude, origin_lat, origin_lon, highway
        )

        # Construir hipóteses em UM loop:
        # - stay_candidates: sair de onde está e atender **agora** (sem esperar) e cabendo na capacidade
        # - depot_candidates: voltar ao depósito (somando retorno), depois ir ao cliente (pode haver espera; capacidade resetada)
        stay_candidates: list[dict] = []
        depot_candidates: list[dict] = []

        for delivery in pending_deliveries:
            # ---- Hipótese A (ficar e ir direto): só vale se couber e sem espera
            if delivery.delivery_weight_tons <= current_vehicle_remaining_capacity:
                _edges_a, distance_a_meters, travel_a_seconds = graph.astar_with_time_between(
                    current_latitude, current_longitude,
                    delivery.delivery_latitude, delivery.delivery_longitude,
                    highway
                )
                arrival_time_seconds_a = current_time_seconds_of_day + travel_a_seconds
                service_time_seconds_a = delivery.earliest_same_day_service_time(arrival_time_seconds_a)
                if service_time_seconds_a is not None:
                    waiting_time_seconds_a = max(0.0, service_time_seconds_a - arrival_time_seconds_a)
                    if waiting_time_seconds_a == 0.0:
                        stay_candidates.append({
                            "delivery": delivery,
                            "distance_meters": float(distance_a_meters),
                            "travel_time_seconds": float(travel_a_seconds),
                            "service_time_seconds": float(service_time_seconds_a),
                        })

            # ---- Hipótese B (voltar ao depósito e planejar): capacidade será resetada; pode esperar
            _edges_b, distance_b_meters, travel_b_seconds = graph.astar_with_time_between(
                origin_lat, origin_lon,
                delivery.delivery_latitude, delivery.delivery_longitude,
                highway
            )
            arrival_time_seconds_b = current_time_seconds_of_day + travel_seconds_back_to_depot + travel_b_seconds
            service_time_seconds_b = delivery.earliest_same_day_service_time(arrival_time_seconds_b)
            if service_time_seconds_b is not None:
                depot_candidates.append({
                    "delivery": delivery,
                    "distance_meters": float(distance_b_meters),
                    "travel_time_seconds": float(travel_b_seconds),
                    "service_time_seconds": float(service_time_seconds_b),
                    "service_time_from_now_seconds": float(service_time_seconds_b - current_time_seconds_of_day),
                })

        # Caso 1: existe pelo menos um candidato “ficar” (imediato e cabe)
        if stay_candidates:
            stay_candidates.sort(
                key=lambda c: (c["service_time_seconds"], c["distance_meters"], c["delivery"].delivery_identifier)
            )
            chosen = stay_candidates[0]
            chosen_delivery: Delivery = chosen["delivery"]

            origin_text = ("da distribuidora"
                           if (current_latitude == origin_lat and current_longitude == origin_lon)
                           else f"de ({current_latitude:.6f}, {current_longitude:.6f})")
            travel_minutes, travel_seconds_remainder = _split_minutes_seconds_exact(chosen["travel_time_seconds"])
            print(
                f"Carga sai {origin_text} {_seconds_to_hhmmss(current_time_seconds_of_day)} "
                f"em direção a ({chosen_delivery.delivery_latitude:.6f}, {chosen_delivery.delivery_longitude:.6f}) "
                f"e chegará em {travel_minutes} minutos {_fmt_seconds(travel_seconds_remainder)} segundos "
                f"às {_seconds_to_hhmmss(chosen['service_time_seconds'])} horas."
            )

            current_time_seconds_of_day = float(chosen["service_time_seconds"])
            current_latitude, current_longitude = chosen_delivery.delivery_latitude, chosen_delivery.delivery_longitude
            current_vehicle_remaining_capacity -= chosen_delivery.delivery_weight_tons
            chosen_delivery.delivery_was_completed = True
            continue

        # Caso 2: não há como atender “agora”; tentar “voltar ao depósito e planejar”
        if depot_candidates:
            # Efetiva o retorno (só imprime e soma tempo se de fato estava fora do depósito)
            if travel_seconds_back_to_depot > 0.0:
                current_time_seconds_of_day += travel_seconds_back_to_depot
                print("Carga retorna a distribuidora.")
                current_latitude, current_longitude = float(origin_lat), float(origin_lon)
                current_vehicle_remaining_capacity = float(capacity_tons_L)

            depot_candidates.sort(
                key=lambda c: (c["service_time_seconds"], c["distance_meters"], c["delivery"].delivery_identifier)
            )
            chosen = depot_candidates[0]
            chosen_delivery: Delivery = chosen["delivery"]

            travel_minutes, travel_seconds_remainder = _split_minutes_seconds_exact(chosen["travel_time_seconds"])
            print(
                f"Carga sai da distribuidora {_seconds_to_hhmmss(current_time_seconds_of_day)} "
                f"em direção a ({chosen_delivery.delivery_latitude:.6f}, {chosen_delivery.delivery_longitude:.6f}) "
                f"e chegará em {travel_minutes} minutos {_fmt_seconds(travel_seconds_remainder)} segundos "
                f"às {_seconds_to_hhmmss(chosen['service_time_seconds'])} horas."
            )

            current_time_seconds_of_day = float(chosen["service_time_seconds"])
            current_latitude, current_longitude = chosen_delivery.delivery_latitude, chosen_delivery.delivery_longitude
            # Após retornar à base a capacidade foi resetada; a entrega sempre cabe (peso ≤ L)
            current_vehicle_remaining_capacity -= chosen_delivery.delivery_weight_tons
            chosen_delivery.delivery_was_completed = True
            continue

        # Caso 3: nada possível HOJE em nenhuma hipótese → fechar o dia
        if travel_seconds_back_to_depot > 0.0:
            current_time_seconds_of_day += travel_seconds_back_to_depot
            print("Carga retorna a distribuidora.")
        current_planning_day += 1
        current_latitude, current_longitude = float(origin_lat), float(origin_lon)
        current_time_seconds_of_day = float(_time_to_seconds(start_time_HH_MM))
        current_vehicle_remaining_capacity = float(capacity_tons_L)
        print(f"\nDIA {current_planning_day}")


# -------------------------------------------------------------------------
# CLIs
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Função: cli_stats
# -------------------------------------------------------------------------
# Descrição
# ---------
# Carrega o grafo a partir dos CSVs e imprime estatísticas básicas: |V| e |E|.
#
# Parâmetros
# ----------
# nodes_csv : str  (caminho para CSV de nós)
# edges_csv : str  (caminho para CSV de arestas)
#
# Retorno
# -------
# None
def cli_stats(nodes_csv: str, edges_csv: str) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    print(f"RoadGraph |V|={graph.node_count()} |E|={graph.edge_count()}")

# -------------------------------------------------------------------------
# Função: cli_node_to_xy_dist
# -------------------------------------------------------------------------
# Descrição
# ---------
# Carrega o grafo e imprime a distância (em metros) do nó <node_id> ao ponto
# (x1, y1). Convenção: x = longitude, y = latitude.
#
# Parâmetros
# ----------
# nodes_csv : str
# edges_csv : str
# x1       : float (longitude do ponto)
# y1       : float (latitude do ponto)
# node_id  : int   (identificador do nó no grafo)
#
# Retorno
# -------
# None (imprime a distância em metros ou a mensagem de ausência do nó)
def cli_node_to_xy_dist(nodes_csv: str, edges_csv: str, x1: float, y1: float, node_id: int) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    if node_id not in graph.nodes:
        print("Não está presente nos arquivos de entrada")
        return
    distance_m = graph.distance_from_node_to_point_m(node_id, float(y1), float(x1))
    print(f"{distance_m:.3f}")


# -------------------------------------------------------------------------
# Função: cli_nearest
# -------------------------------------------------------------------------
# Descrição
# ---------
# Encontra o nó roteável mais próximo às coordenadas informadas, respeitando o
# filtro de modal (car|bike|foot), e imprime o osmid.
#
# Parâmetros
# ----------
# nodes_csv : str
# edges_csv : str
# lat, lon  : float
# highway   : str ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None
def cli_nearest(nodes_csv: str, edges_csv: str, lat: float, lon: float, highway: str) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    print(graph.nearest_node(lat, lon, highway))


# -------------------------------------------------------------------------
# Função: cli_dijkstra
# -------------------------------------------------------------------------
# Descrição
# ---------
# Calcula o caminho mínimo por Dijkstra entre dois pontos (lat/lon) e imprime
# a sequência de arestas com distâncias e tempo estimado por aresta e total.
#
# Parâmetros
# ----------
# nodes_csv         : str
# edges_csv         : str
# lat1, lon1        : float (origem)
# lat2, lon2        : float (destino)
# highway           : str   ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None
def cli_dijkstra(nodes_csv: str, edges_csv: str,
                 lat1: float, lon1: float, lat2: float, lon2: float,
                 highway: str) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    source_node_id = graph.nearest_node(lat1, lon1, highway)
    path_edges = graph.shortest_path_between(lat1, lon1, lat2, lon2, highway)
    if not path_edges:
        print("[]")
        return
    print_path_edges(path_edges, source_node_id, "Dijkstra")


# -------------------------------------------------------------------------
# Função: cli_astar
# -------------------------------------------------------------------------
# Descrição
# ---------
# Calcula o caminho mínimo por A* (distância) entre dois pontos e imprime
# a sequência de arestas com distâncias e tempo estimado.
#
# Parâmetros
# ----------
# nodes_csv         : str
# edges_csv         : str
# lat1, lon1        : float (origem)
# lat2, lon2        : float (destino)
# highway           : str   ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None
def cli_astar(nodes_csv: str, edges_csv: str,
              lat1: float, lon1: float, lat2: float, lon2: float,
              highway: str) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    source_node_id = graph.nearest_node(lat1, lon1, highway)
    path_edges = graph.astar_shortest_path_between(lat1, lon1, lat2, lon2, highway)
    if not path_edges:
        print("[]")
        return
    print_path_edges(path_edges, source_node_id, "A*")


# -------------------------------------------------------------------------
# Função: cli_astar_time
# -------------------------------------------------------------------------
# Descrição
# ---------
# Calcula o caminho por A* minimizando custo artificial e acumulando o
# tempo real de viagem. Imprime a sequência de arestas e o total de
# distância/tempo.
#
# Parâmetros
# ----------
# nodes_csv         : str
# edges_csv         : str
# lat1, lon1        : float (origem)
# lat2, lon2        : float (destino)
# highway           : str   ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None
def cli_astar_time(nodes_csv: str, edges_csv: str,
                   lat1: float, lon1: float, lat2: float, lon2: float,
                   highway: str) -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    source_node_id = graph.nearest_node(lat1, lon1, highway)
    path_edges, total_distance_m, total_time_s = graph.astar_with_time_between(lat1, lon1, lat2, lon2, highway)
    if not path_edges:
        print("[]")
        return
    print_path_edges(path_edges, source_node_id, "A* (com tempo)")


# -------------------------------------------------------------------------
# Função: cli_vrp
# -------------------------------------------------------------------------
# Descrição
# ---------
# Executa a heurística VRP a partir de um arquivo de entrada em texto,
# carregando o grafo a partir dos CSVs e repassando os parâmetros.
#
# Parâmetros
# ----------
# nodes_csv  : str (caminho para CSV de nós)
# edges_csv  : str (caminho para CSV de arestas)
# input_txt  : str (arquivo de entrada com origem/capacidade e entregas)
# start_HH_MM: str (horário de partida "HH:MM")
# highway    : str ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None (imprime o plano de entregas)
def cli_vrp(nodes_csv: str, edges_csv: str,
            input_txt: str, start_HH_MM: str, highway: str = "car") -> None:
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    origin_lat, origin_lon, capacity_tons, deliveries_raw = parse_vrp_input_file(input_txt)
    VRP_heuristic(graph, origin_lat, origin_lon, start_HH_MM, capacity_tons, deliveries_raw, highway)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    prog = os.path.basename(__file__)
    usage = (
        "Uso:\n"
        f"  python {prog} stats <out_nodes_path> <out_edges_path>\n"
        f"  python {prog} nearest <out_nodes_path> <out_edges_path> <lat> <lon> <car|bike|foot>\n"
        f"  python {prog} dijkstra <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} astar  <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} astar_time  <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} vrp <out_nodes_path> <out_edges_path> <input_txt> <start_HH:MM> <car|bike|foot>\n"
        f"  python {prog} node_to_xy_dist <out_nodes_path> <out_edges_path> <x1> <y1> <node_id>\n"
    )
    start_time = time.time()
    if len(sys.argv) == 4 and sys.argv[1] == "stats":
        _, _, nodes_csv, edges_csv = sys.argv
        cli_stats(nodes_csv, edges_csv)
    elif len(sys.argv) == 7 and sys.argv[1] == "nearest":
        _, _, nodes_csv, edges_csv, lat, lon = sys.argv[:6]
        cli_nearest(nodes_csv, edges_csv, float(lat), float(lon), sys.argv[6])
    elif len(sys.argv) == 9 and sys.argv[1] == "dijkstra":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_dijkstra(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 9 and sys.argv[1] == "astar":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_astar(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 9 and sys.argv[1] == "astar_time":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_astar_time(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 7 and sys.argv[1] == "vrp":
        _, _, nodes_csv, edges_csv, input_txt, start_hhmm = sys.argv[:6]
        cli_vrp(nodes_csv, edges_csv, input_txt, start_hhmm, sys.argv[6])
    elif len(sys.argv) == 7 and sys.argv[1] == "node_to_xy_dist":
        _, _, nodes_csv, edges_csv, x1, y1 = sys.argv[:6]
        node_id = int(sys.argv[6])
        cli_node_to_xy_dist(nodes_csv, edges_csv, float(x1), float(y1), node_id)
    else:
        print(usage, end="")
    execution_time = time.time() - start_time
    print("TTempo de execução total: %s segundos." % execution_time)
