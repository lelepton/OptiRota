# -*- coding: utf-8 -*-
"""
optirota.py
-----------
Refatoração focada em:
1) Reutilização de funções utilitárias (Haversine e filtro de arestas).
2) Eliminação de duplicação de constantes/lógica (coeficientes e velocidades).
3) Correções em chamadas de CLI (load_graph → RoadGraph.load_graph).
4) Comentários/documentação uniformes no mesmo estilo do arquivo original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import time
import pandas as pd
import math
import os
import sys
import json

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
                              highway: str) -> tuple[list[Edge], float]:
        
        """
        Executa Dijkstra entre dois pares (lat, lon), restringindo às arestas
        válidas para o 'highway' informado.

        Retorna
        -------
        (path_edges, total_distance_m) onde:
          - path_edges : list[Edge] na ordem origem→destino (vazia se não houver caminho)
          - total_distance_m : float com a distância total do menor caminho (0.0 se vazio)
        """
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
            return [], 0.0

        path_edges = self._reconstruct_path(predecessor, source_node_id, target_node_id)
        total_distance_m = sum(e.w for e in path_edges)
        return path_edges, total_distance_m

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
                                    highway: str) -> tuple[list[Edge], float, float]:
        
        """
        Executa A* (custo = distância) entre (lat1,lon1) e (lat2,lon2).

        Retorna
        -------
        (path_edges, total_distance_m, total_time_s) onde:
          - path_edges       : list[Edge] na ordem origem→destino (vazia se não houver caminho)
          - total_distance_m : float com a distância total do caminho
          - total_time_s     : float com o tempo estimado (somando w/speed por aresta)
        """
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
            return [], 0.0, float("inf")

        path_edges = self._reconstruct_path(predecessor, source_node_id, target_node_id)
        total_distance_m = sum(e.w for e in path_edges)
        total_time_s = sum(e.w / _speed_mps_for_tag(e.tag) for e in path_edges) if path_edges else 0.0
        return path_edges, total_distance_m, total_time_s

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
# Substitua a função parse_vrp_input_file no optirota.py por esta versão corrigida

# Substitua a função parse_vrp_input_file no optirota.py por esta versão

def parse_vrp_input_file(file_path: str) -> tuple[float, float, list[float], list[list]]:
    """
    Parser robusto para arquivo VRP.
    
    Formato esperado:
      Linha 1: lat,lon,num_veiculos
      Linha 2: capacidade_por_veiculo (valor único ou lista)
      Linhas 3+: lat,lon,peso_toneladas,HH:MM-HH:MM[,HH:MM-HH:MM]
    
    Exemplo:
      -9.660217,-35.718473,10
      3
      -9.658953,-35.706859,2,08:00-18:00
      -9.662428,-35.700250,1.5,08:00-18:00
    """
    import re

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    # Remove vazias/puras de comentário
    lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) < 2:
        raise ValueError("Arquivo VRP precisa de ao menos 2 linhas (origem+veículos e capacidades).")

    # ===== LINHA 1: Origem e número de veículos =====
    line1_normalized = lines[0].replace(",", " ")
    tokens1 = [t.strip() for t in line1_normalized.split() if t.strip()]
    
    if len(tokens1) < 3:
        raise ValueError(f"Linha 1 inválida: {lines[0]!r}. Esperado: <lat> <lon> <num_veiculos>")
    
    try:
        origin_lat = float(tokens1[0])
        origin_lon = float(tokens1[1])
        num_vehicles = int(tokens1[2])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Linha 1 inválida: {lines[0]!r}. Esperado: <lat> <lon> <num_veiculos>") from e

    # ===== LINHA 2: Capacidades =====
    # Pode ser um único valor (replicado para N veículos) ou lista de N valores
    caps_tokens = re.split(r'[,\s]+', lines[1].strip())
    caps_tokens = [t for t in caps_tokens if t]
    
    try:
        if len(caps_tokens) == 1:
            # Capacidade única: usar para todos os veículos
            cap_value = float(caps_tokens[0])
            capacities_tons = [cap_value] * num_vehicles
        else:
            # Lista de capacidades
            capacities_tons = [float(t) for t in caps_tokens[:num_vehicles]]
            if len(capacities_tons) < num_vehicles:
                # Preencher com o último valor se faltarem
                last_cap = capacities_tons[-1] if capacities_tons else 3.0
                capacities_tons.extend([last_cap] * (num_vehicles - len(capacities_tons)))
    except ValueError as e:
        raise ValueError(f"Linha 2 inválida: {lines[1]!r}. Esperado: capacidade(s) em toneladas") from e

    # ===== LINHAS 3+: Entregas =====
    deliveries_raw: list[list] = []
    for idx, line in enumerate(lines[2:], start=3):
        try:
            # Divide por vírgula
            parts = [p.strip() for p in line.split(",")]
            
            if len(parts) < 4:
                raise ValueError(f"Número insuficiente de campos (esperado >=4, recebido {len(parts)})")
            
            # Sempre: lat, lon, peso, janela
            lat = float(parts[0])
            lon = float(parts[1])
            peso = float(parts[2])
            
            # Janelas (pode ser 1 ou mais, separadas por vírgula)
            # Junta tudo após o peso como string de janelas
            windows_string = ",".join(parts[3:]).strip()
            
            if not windows_string:
                raise ValueError("Janela de tempo vazia")
            
            deliveries_raw.append([lat, lon, peso, windows_string])
            
        except ValueError as exc:
            raise ValueError(f"Linha {idx} inválida: {line!r} | erro: {exc}") from exc

    return origin_lat, origin_lon, capacities_tons, deliveries_raw
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

# -------------------------------------------------------------------------
# Função: VRP_heuristic
# -------------------------------------------------------------------------
# Descrição
# ---------
# Heurística gulosa com janelas diárias e **vários veículos**.
# Mudança: antes de empurrar entrega para "DIA N+1", tentar alocar no
# MESMO dia para outro veículo (imprimindo "DIA N, VEÍCULO M").
#
# Parâmetros
# ----------
# origin_lat       : float  (latitude do depósito)
# origin_lon       : float  (longitude do depósito)
# start_time_HH_MM : str    (horário de partida "HH:MM")
# capacities_tons_L: list   (capacidade de cada veículo, em toneladas)
# deliveries_raw   : list   ([[lat, lon, weight_tons, windows_string], ...])
# highway          : str    ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# None (imprime o plano por "DIA N, VEÍCULO M")

# -------------------------------------------------------------------------
# Função: VRP_heuristic
# -------------------------------------------------------------------------
# Descrição
# ---------
# Heurística gulosa com janelas diárias e vários veículos.
# Mudança: quando termina o bloco "DIA N, VEÍCULO M" (troca de veículo,
# mudança de dia ou fim do plano), imprime imediatamente:
#   Carga retorna a distribuidora.
#
# Parâmetros
# ----------
# origin_lat       : float
# origin_lon       : float
# start_time_HH_MM : str
# capacities_tons_L: list[float]
# deliveries_raw   : list[[lat, lon, weight_tons, windows_string]]
# highway          : str
#
# Retorno
# -------
# None
# -------------------------------------------------------------------------
# Função: VRP_heuristic
# -------------------------------------------------------------------------
# Descrição
# ---------
# Heurística gulosa com janelas diárias e vários veículos.
# Antes de empurrar entrega para “DIA N+1”, tenta alocar no MESMO dia
# para outro veículo (imprimindo “DIA N, VEÍCULO M” quando o veículo
# começa a operar no dia).
#
# Mudança de interface (RETORNO):
# -------------------------------
# Retorna uma lista: [(veiculo_id_1based, lista_de_arestas_percorridas), ...]
# - veiculo_id_1based : int (1, 2, 3, ...)
# - lista_de_arestas_percorridas : List[Edge] concatenando, na ordem,
#   as arestas dos deslocamentos realizados por aquele veículo.
#
# Parâmetros
# ----------
# graph              : RoadGraph
# origin_lat         : float  (latitude do depósito)
# origin_lon         : float  (longitude do depósito)
# start_time_HH_MM   : str    (horário de partida “HH:MM”)
# capacities_tons_L  : list   (capacidade [t] de cada veículo)
# deliveries_raw     : list   ([[lat, lon, weight_tons, windows_str], ...])
# highway            : str    ('car' | 'bike' | 'foot')
#
# Retorno
# -------
# List[Tuple[int, List[Edge]]] : (veículo 1-based, arestas)
def VRP_heuristic(
    graph: "RoadGraph",
    origin_lat: float,
    origin_lon: float,
    start_time_HH_MM: str,
    capacities_tons_L: list[float],
    deliveries_raw: list[list],
    highway: str = "car",
) -> list[tuple[int, list["Edge"]]]:
    # 1) Construção das entregas
    deliveries: list[Delivery] = [
        Delivery.from_windows_string(idx, lat, lon, windows_string, weight_tons)
        for idx, (lat, lon, weight_tons, windows_string) in enumerate(deliveries_raw)
    ]

    # 2) Estado dos veículos
    num_vehicles = len(capacities_tons_L)
    vehicles = [{
        "lat": float(origin_lat),
        "lon": float(origin_lon),
        "t": float(_time_to_seconds(start_time_HH_MM)),
        "cap": float(capacities_tons_L[i]),
        "cap_max": float(capacities_tons_L[i]),
        "started": False,  # já imprimiu "DIA N, VEÍCULO M" no dia atual
    } for i in range(num_vehicles)]

    # 2.1) Acumulador das arestas percorridas por veículo
    vehicles_paths: list[list[Edge]] = [[] for _ in range(num_vehicles)]

    current_day = 1
    print(f"DIA {current_day}")
    delivered_today = False
    current_block_vi: int | None = None  # veículo cujo bloco está “aberto”

    # ---------------------------------------------------------------------
    # Função interna: melhor candidato para UM veículo
    # ---------------------------------------------------------------------
    # Retorna um dict com:
    #   - delivery
    #   - distance_meters
    #   - travel_time_seconds
    #   - service_time_seconds
    #   - edges              (se “direct”)
    #   - edges_return       (se “via depósito”)
    #   - edges_to_delivery  (se “via depósito”)
    # e também o tipo: "direct" | "depot"
    def best_candidate_for_vehicle(vstate: dict) -> tuple[Optional[dict], Optional[str]]:
        cur_lat = vstate["lat"]; cur_lon = vstate["lon"]; cur_t = vstate["t"]; cur_cap = vstate["cap"]

        direct_candidates: list[dict] = []
        depot_candidates:  list[dict] = []

        # Caminho de retorno atual->depósito (para calcular tempo de volta quando “via depósito”)
        edges_back, _dist_back, ret_back_s = graph.astar_with_time_between(
            cur_lat, cur_lon, origin_lat, origin_lon, highway
        )

        for delivery in deliveries:
            if delivery.delivery_was_completed:
                continue

            # A) Direto (sem voltar ao depósito)
            if delivery.delivery_weight_tons <= cur_cap:
                edges_a, dist_a, trav_a = graph.astar_with_time_between(
                    cur_lat, cur_lon, delivery.delivery_latitude, delivery.delivery_longitude, highway
                )
                svc_a = delivery.earliest_same_day_service_time(cur_t + trav_a)
                if svc_a is not None:
                    direct_candidates.append({
                        "delivery": delivery,
                        "distance_meters": float(dist_a),
                        "travel_time_seconds": float(trav_a),
                        "service_time_seconds": float(svc_a),
                        "edges": edges_a,
                    })

            # B) Via depósito (reset de capacidade)
            edges_b, dist_b, trav_b = graph.astar_with_time_between(
                origin_lat, origin_lon, delivery.delivery_latitude, delivery.delivery_longitude, highway
            )
            svc_b = delivery.earliest_same_day_service_time(cur_t + ret_back_s + trav_b)
            if svc_b is not None and delivery.delivery_weight_tons <= vstate["cap_max"]:
                depot_candidates.append({
                    "delivery": delivery,
                    "distance_meters": float(dist_b),
                    "travel_time_seconds": float(trav_b),
                    "service_time_seconds": float(svc_b),
                    "edges_return": edges_back,
                    "edges_to_delivery": edges_b,
                })

        def _rank(c: dict) -> tuple:
            return (c["service_time_seconds"], c["distance_meters"], c["delivery"].delivery_identifier)

        if direct_candidates:
            direct_candidates.sort(key=_rank)
            return direct_candidates[0], "direct"
        if depot_candidates:
            depot_candidates.sort(key=_rank)
            return depot_candidates[0], "depot"
        return None, None

    # ---------------------------------------------------------------------
    # Loop principal (dias)
    # ---------------------------------------------------------------------
    while True:
        pending = [d for d in deliveries if not d.delivery_was_completed]
        if not pending:
            # Fechar bloco ativo (se houver), antes de encerrar
            if current_block_vi is not None:
                print("Carga retorna a distribuidora.")
                current_block_vi = None
            # Fim → retorna estrutura solicitada
            return [(vi + 1, vehicles_paths[vi]) for vi in range(num_vehicles)]

        # Melhor atendimento entre todos os veículos
        best = None; best_vi = None; best_kind = None
        for vi, v in enumerate(vehicles):
            cand, kind = best_candidate_for_vehicle(v)
            if cand is None:
                continue
            key = (cand["service_time_seconds"], cand["distance_meters"], cand["delivery"].delivery_identifier)
            if best is None or key < (best["service_time_seconds"], best["distance_meters"], best["delivery"].delivery_identifier):
                best, best_vi, best_kind = cand, vi, kind

        if best is not None:
            # Troca de bloco de veículo: fecha o anterior
            if current_block_vi is None:
                current_block_vi = best_vi
            elif current_block_vi != best_vi:
                print("Carga retorna a distribuidora.")
                current_block_vi = best_vi

            v = vehicles[best_vi]
            d = best["delivery"]

            # Cabeçalho do bloco do veículo no dia
            if not v["started"]:
                print(f"DIA {current_day}, VEÍCULO {best_vi + 1}")
                v["started"] = True

            # -------------------------------------------------------------
            # Caso A: direto
            # -------------------------------------------------------------
            if best_kind == "direct":
                mins, rem = _split_minutes_seconds_exact(best["travel_time_seconds"])
                print(
                    f"Carga sai ({v['lat']:.6f}, {v['lon']:.6f}) {_seconds_to_hhmmss(v['t'])} "
                    f"em direção a ({d.delivery_latitude:.6f}, {d.delivery_longitude:.6f}) "
                    f"e chegará em {mins} minutos {_fmt_seconds(rem)} segundos "
                    f"às {_seconds_to_hhmmss(best['service_time_seconds'])} horas."
                )
                # Acumula arestas do deslocamento direto
                vehicles_paths[best_vi].extend(best.get("edges", []))

                v["t"] = float(best["service_time_seconds"])
                v["lat"], v["lon"] = d.delivery_latitude, d.delivery_longitude
                v["cap"] -= d.delivery_weight_tons
                d.delivery_was_completed = True
                delivered_today = True
                continue

            # -------------------------------------------------------------
            # Caso B: via depósito (reset capacidade)
            # -------------------------------------------------------------
            # Acumula arestas: retorno e ida ao cliente
            vehicles_paths[best_vi].extend(best.get("edges_return", []))
            mins, rem = _split_minutes_seconds_exact(best["travel_time_seconds"])
            print(
                f"Carga sai da distribuidora {_seconds_to_hhmmss(v['t'])} "
                f"em direção a ({d.delivery_latitude:.6f}, {d.delivery_longitude:.6f}) "
                f"e chegará em {mins} minutos {_fmt_seconds(rem)} segundos "
                f"às {_seconds_to_hhmmss(best['service_time_seconds'])} horas."
            )
            vehicles_paths[best_vi].extend(best.get("edges_to_delivery", []))

            v["t"] = float(best["service_time_seconds"])
            v["lat"], v["lon"] = d.delivery_latitude, d.delivery_longitude
            v["cap"] -= d.delivery_weight_tons
            # reset de cap já é implícito ao voltar no cálculo; aqui seguimos a lógica original:
            # (se desejar “zerar” explicitamente na volta, mover esse reset para o ponto de retorno)
            d.delivery_was_completed = True
            delivered_today = True
            continue

        # Nenhum veículo atende algo hoje → checa inviabilidade e vira o dia
        if not delivered_today:
            possible_next_day = False
            day_start_seconds = float(_time_to_seconds(start_time_HH_MM))
            max_caps = [vv["cap_max"] for vv in vehicles] or [0.0]
            for dcheck in [d for d in deliveries if not d.delivery_was_completed]:
                if dcheck.delivery_weight_tons > max(max_caps):
                    print(f"Entrega {dcheck.delivery_identifier} inviável: peso maior que a capacidade de qualquer veículo.")
                    dcheck.delivery_was_completed = True
                    continue
                _eX, _dX, travX = graph.astar_with_time_between(
                    origin_lat, origin_lon, dcheck.delivery_latitude, dcheck.delivery_longitude, highway
                )
                svcX = dcheck.earliest_same_day_service_time(day_start_seconds + travX)
                if svcX is not None:
                    possible_next_day = True
                    break
            if not possible_next_day:
                if current_block_vi is not None:
                    print("Carga retorna a distribuidora.")
                    current_block_vi = None
                print("Nenhuma entrega restante é atendível em dias futuros (por capacidade/janelas/alcance). Encerrando.")
                return [(vi + 1, vehicles_paths[vi]) for vi in range(num_vehicles)]

        # Fechar bloco ativo antes de virar o dia
        if current_block_vi is not None:
            print("Carga retorna a distribuidora.")
            current_block_vi = None

        # Novo dia: reseta posição/tempo/capacidade de todos
        current_day += 1
        print(f"DIA {current_day}")
        delivered_today = False
        for v in vehicles:
            v["lat"], v["lon"] = float(origin_lat), float(origin_lon)
            v["t"] = float(_time_to_seconds(start_time_HH_MM))
            v["cap"] = float(v["cap_max"])
            v["started"] = False


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
    path_edges, total_distance_m = graph.shortest_path_between(lat1, lon1, lat2, lon2, highway)
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
    path_edges, total_distance_m, total_time_s = graph.astar_shortest_path_between(lat1, lon1, lat2, lon2, highway)
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
# Função: save_routes_to_json
# -------------------------------------------------------------------------
# Descrição
# ---------
# Salva as rotas retornadas pelo VRP_heuristic em um arquivo JSON
# com as coordenadas de cada segmento para visualização.
#
# Parâmetros
# ----------
# routes       : list[tuple(int, list[Edge])] - saída do VRP_heuristic
# graph        : RoadGraph - grafo com informações de nós
# output_file  : str - caminho do arquivo JSON de saída
#
# Retorno
# -------
# None
def save_routes_to_json(routes: list[tuple[int, list["Edge"]]], 
                        graph: "RoadGraph", 
                        output_file: str) -> None:
    """
    Converte rotas (com arestas) em coordenadas para JSON.
    Formato: {
        "routes": [
            {
                "vehicle_id": 1,
                "segments": [
                    {
                        "from": [lat, lon],
                        "to": [lat, lon],
                        "distance_m": 150.5,
                        "highway": "residential"
                    },
                    ...
                ]
            }
        ]
    }
    """
    routes_data = {"routes": []}
    
    for vehicle_id, edges in routes:
        segments = []
        
        for edge in edges:
            # Encontrar nó de origem (procura em todas as arestas para achar quem tem este destino)
            # Estratégia: usar o nó anterior na sequência
            from_node_id = None
            
            # Busca no grafo pelo nó que tem esta aresta como saída
            for node_id, outgoing in graph.adj.items():
                for outgoing_edge in outgoing:
                    if (outgoing_edge.v == edge.v and 
                        outgoing_edge.w == edge.w and 
                        outgoing_edge.tag == edge.tag):
                        from_node_id = node_id
                        break
                if from_node_id:
                    break
            
            if from_node_id and from_node_id in graph.nodes and edge.v in graph.nodes:
                from_lat, from_lon = graph.nodes[from_node_id]
                to_lat, to_lon = graph.nodes[edge.v]
                
                segment = {
                    "from": [float(from_lat), float(from_lon)],
                    "to": [float(to_lat), float(to_lon)],
                    "distance_m": float(edge.w),
                    "highway": edge.tag,
                    "name": edge.name
                }
                segments.append(segment)
        
        route_info = {
            "vehicle_id": int(vehicle_id),
            "num_segments": len(segments),
            "total_distance_m": sum(s["distance_m"] for s in segments),
            "segments": segments
        }
        routes_data["routes"].append(route_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(routes_data, f, indent=2, ensure_ascii=False)

# -------------------------------------------------------------------------
# Função: load_routes_from_json
# -------------------------------------------------------------------------
# Descrição
# ---------
# Carrega as rotas salvas em JSON.
#
# Parâmetros
# ----------
# input_file : str - caminho do arquivo JSON
#
# Retorno
# -------
# dict - estrutura com as rotas
def load_routes_from_json(input_file: str) -> dict:
    """Carrega rotas de um arquivo JSON."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"routes": []}


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
    from pathlib import Path
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    origin_lat, origin_lon, capacities_tons_L, deliveries_raw = parse_vrp_input_file(input_txt)
    
    # Executar VRP e capturar rotas
    routes = VRP_heuristic(graph, origin_lat, origin_lon, start_HH_MM, 
                          capacities_tons_L, deliveries_raw, highway)
    
    # Salvar rotas em JSON para visualização
    output_dir = Path(input_txt).parent
    routes_json = output_dir / "vrp_routes.json"
    save_routes_to_json(routes, graph, str(routes_json))
    print(f"Rotas salvas em: {routes_json}")


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
