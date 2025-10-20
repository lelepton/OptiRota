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
import re

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
    return SPEED_LIMITS_MPS.get(_normalize_tag(tag), 8.33) # Padrão para residential


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
    return COEF_BY_TAG.get(_normalize_tag(tag), 0.3) # Padrão para residential


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
        
        if not candidate_node_ids:
             # Fallback: se nenhum nó for roteável, procura em todos os nós
             candidate_node_ids = set(self.nodes.keys())

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
        # CORREÇÃO: trocado \u2248 por ~ para evitar UnicodeEncodeError no Windows
        print(f"{running_source} -> {edge.v} | w={edge.w:.3f} m | t~{edge_time_fmt} | cumulativo={cumulative_distance:.3f} m | tag={edge.tag} | name=\"{edge.name}\"")
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
# CORREÇÃO: Função simplificada para evitar erros de parsing.
def parse_vrp_input_file(file_path: str) -> tuple[float, float, list[float], list[list]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) < 2:
        raise ValueError("Arquivo VRP precisa de ao menos 2 linhas (origem+veículos e capacidades).")

    # Linha 1: Origem e número de veículos
    tokens1 = [t.strip() for t in lines[0].replace(",", " ").split() if t.strip()]
    if len(tokens1) < 3:
        raise ValueError(f"Linha 1 inválida: {lines[0]!r}. Esperado: <lat> <lon> <num_veiculos>")
    
    try:
        origin_lat = float(tokens1[0])
        origin_lon = float(tokens1[1])
        num_vehicles = int(tokens1[2])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Linha 1 inválida: {lines[0]!r}. Erro: {e}") from e

    # Linha 2: Capacidades
    caps_tokens = re.split(r'[,\s]+', lines[1].strip())
    caps_tokens = [t for t in caps_tokens if t]
    
    try:
        if len(caps_tokens) == 1:
            cap_value = float(caps_tokens[0])
            capacities_tons = [cap_value] * num_vehicles
        else:
            capacities_tons = [float(t) for t in caps_tokens[:num_vehicles]]
            if len(capacities_tons) < num_vehicles:
                last_cap = capacities_tons[-1] if capacities_tons else 3.0
                capacities_tons.extend([last_cap] * (num_vehicles - len(capacities_tons)))
    except (ValueError, IndexError) as e:
        raise ValueError(f"Linha 2 inválida: {lines[1]!r}. Erro: {e}") from e

    # Linhas 3+: Entregas
    deliveries_raw: list[list] = []
    for idx, line in enumerate(lines[2:], start=3):
        try:
            # Divide a linha por vírgula e remove espaços extras
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 4:
                # Tenta dividir por espaço como fallback
                parts = [p.strip() for p in line.split() if p.strip()]
                if len(parts) < 4:
                    raise ValueError(f"Linha de entrega deve ter pelo menos 4 campos. Recebido: {len(parts)}")
            
            lat = float(parts[0])
            lon = float(parts[1])
            weight = float(parts[2])
            # Junta todas as partes restantes como a string de janelas
            windows_string = ",".join(parts[3:]).strip().strip('"')

            if not windows_string:
                raise ValueError("Janela de tempo não encontrada ou vazia.")
            
            deliveries_raw.append([lat, lon, weight, windows_string])
            
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Linha {idx} inválida: {line!r} | erro: {exc}") from exc

    return origin_lat, origin_lon, capacities_tons, deliveries_raw

# -------------------------------------------------------------------------
# Função: VRP_heuristic (NOVA VERSÃO MULTI-STOP)
# -------------------------------------------------------------------------
def VRP_heuristic(
    graph: "RoadGraph",
    origin_lat: float,
    origin_lon: float,
    start_time_HH_MM: str,
    capacities_tons_L: list[float],
    deliveries_raw: list[list],
    highway: str = "car",
) -> list[tuple[int, list["Edge"]]]:
    
    deliveries: list[Delivery] = [
        Delivery.from_windows_string(idx, lat, lon, win_str, weight)
        for idx, (lat, lon, weight, win_str) in enumerate(deliveries_raw)
    ]
    
    num_vehicles = len(capacities_tons_L)
    vehicles_paths: list[list[Edge]] = [[] for _ in range(num_vehicles)]
    
    current_day = 1
    
    while any(not d.delivery_was_completed for d in deliveries):
        print(f"DIA {current_day}")
        
        # Estado dos veículos para o dia atual
        vehicles = [{
            "lat": origin_lat, "lon": origin_lon,
            "t": float(_time_to_seconds(start_time_HH_MM)),
            "cap": capacities_tons_L[i], "cap_max": capacities_tons_L[i],
            "started_day": False, "tour_path": []
        } for i in range(num_vehicles)]
        
        day_had_deliveries = False

        for vi in range(num_vehicles):
            v = vehicles[vi]
            
            while True: # Loop para construir um tour completo para o veículo vi
                
                # Encontrar a melhor PRÓXIMA entrega para ESTE veículo
                candidates = []
                for delivery in deliveries:
                    if delivery.delivery_was_completed or delivery.delivery_weight_tons > v["cap"]:
                        continue

                    # Calcular tempo de viagem da pos. atual do veículo até a entrega
                    edges, dist, travel_time = graph.astar_with_time_between(
                        v["lat"], v["lon"],
                        delivery.delivery_latitude, delivery.delivery_longitude,
                        highway
                    )
                    
                    # Verificar se a entrega é possível hoje
                    arrival_time = v["t"] + travel_time
                    service_time = delivery.earliest_same_day_service_time(arrival_time)
                    
                    if service_time is not None:
                        candidates.append({
                            "delivery": delivery,
                            "edges": edges,
                            "service_time": service_time,
                            "travel_time": travel_time
                        })
                
                if not candidates:
                    # Nenhuma entrega mais pode ser feita por este veículo nesta viagem
                    if v["started_day"]:
                         print("Carga retorna a distribuidora.")
                    break # Fim do tour, passa para o próximo veículo

                # Critério de seleção: menor tempo de chegada (serviço)
                best_cand = min(candidates, key=lambda c: c["service_time"])
                d = best_cand["delivery"]
                
                # Imprimir cabeçalho do veículo se for a primeira entrega do dia
                if not v["started_day"]:
                    print(f"DIA {current_day}, VEÍCULO {vi + 1}")
                    v["started_day"] = True
                    day_had_deliveries = True

                # Imprimir log de deslocamento
                mins, rem = _split_minutes_seconds_exact(best_cand["travel_time"])
                print(
                    f"Carga sai ({v['lat']:.6f}, {v['lon']:.6f}) {_seconds_to_hhmmss(v['t'])} "
                    f"em direção a ({d.delivery_latitude:.6f}, {d.delivery_longitude:.6f}) "
                    f"e chegará em {mins} minutos {_fmt_seconds(rem)} segundos "
                    f"às {_seconds_to_hhmmss(best_cand['service_time'])} horas."
                )

                # Atualizar estado do veículo
                v["t"] = best_cand["service_time"]
                v["lat"], v["lon"] = d.delivery_latitude, d.delivery_longitude
                v["cap"] -= d.delivery_weight_tons
                vehicles_paths[vi].extend(best_cand["edges"])
                
                # Marcar entrega como concluída
                d.delivery_was_completed = True

        if not day_had_deliveries and any(not d.delivery_was_completed for d in deliveries):
             print("Nenhuma entrega pôde ser realizada hoje. Verificando inviabilidade...")
             # Lógica para verificar se alguma entrega restante é impossível
             possible_in_future = False
             for d in deliveries:
                 if not d.delivery_was_completed:
                     if any(d.delivery_weight_tons <= cap for cap in capacities_tons_L):
                         possible_in_future = True
                         break
             if not possible_in_future:
                 print("Entregas restantes são inviáveis por capacidade. Encerrando.")
                 break

        current_day += 1

    return [(vi + 1, path) for vi, path in enumerate(vehicles_paths)]


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
    
    # Salvar resultado em JSON para o visualizador
    from pathlib import Path
    output_dir = Path(nodes_csv).parent
    path_route_json = output_dir / "path_route.json"
    save_single_path_to_json([(1, path_edges)], graph, str(path_route_json), source_node_id)
    
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

    # Salvar resultado em JSON para o visualizador
    from pathlib import Path
    output_dir = Path(nodes_csv).parent
    path_route_json = output_dir / "path_route.json"
    save_single_path_to_json([(1, path_edges)], graph, str(path_route_json), source_node_id)

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
    
    # Salvar resultado em JSON para o visualizador
    from pathlib import Path
    output_dir = Path(nodes_csv).parent
    path_route_json = output_dir / "path_route.json"
    save_single_path_to_json([(1, path_edges)], graph, str(path_route_json), source_node_id)

    if not path_edges:
        print("[]")
        return
    print_path_edges(path_edges, source_node_id, "A* (com tempo)")

# -------------------------------------------------------------------------
# Função: save_routes_to_json (Genérica para VRP e Pathfinding)
# -------------------------------------------------------------------------
def save_routes_to_json(routes: list[tuple[int, list["Edge"]]], 
                        graph: "RoadGraph", 
                        output_file: str,
                        # Para pathfinding, o nó inicial é conhecido
                        initial_source_node_id: Optional[int] = None) -> None:
    routes_data = {"routes": []}
    
    for vehicle_id, edges in routes:
        if not edges: continue
        
        from_node_id = initial_source_node_id
        if from_node_id is None:
            # Lógica para VRP: Encontra o nó de origem da primeira aresta
            first_edge = edges[0]
            found = False
            for node_id, outgoing_edges in graph.adj.items():
                if first_edge in outgoing_edges:
                    from_node_id = node_id
                    found = True
                    break
            if not found: continue
        
        segments = []
        current_node_id = from_node_id
        for edge in edges:
            if current_node_id not in graph.nodes or edge.v not in graph.nodes:
                continue
            from_lat, from_lon = graph.nodes[current_node_id]
            to_lat, to_lon = graph.nodes[edge.v]
            
            segment = {
                "from": [float(from_lat), float(from_lon)],
                "to": [float(to_lat), float(to_lon)],
                "distance_m": float(edge.w),
                "highway": edge.tag,
                "name": edge.name
            }
            segments.append(segment)
            current_node_id = edge.v
        
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
# Função: save_single_path_to_json (Wrapper para Dijkstra/A*)
# -------------------------------------------------------------------------
def save_single_path_to_json(routes: list[tuple[int, list["Edge"]]], 
                            graph: "RoadGraph", 
                            output_file: str,
                            source_node_id: int) -> None:
    save_routes_to_json(routes, graph, output_file, initial_source_node_id=source_node_id)


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
    if len(sys.argv) < 2:
        print(usage, end="")
        sys.exit(1)
        
    command = sys.argv[1]

    try:
        if command == "stats" and len(sys.argv) == 4:
            _, _, nodes_csv, edges_csv = sys.argv
            cli_stats(nodes_csv, edges_csv)
        elif command == "nearest" and len(sys.argv) == 7:
            _, _, nodes_csv, edges_csv, lat, lon, transport = sys.argv
            cli_nearest(nodes_csv, edges_csv, float(lat), float(lon), transport)
        elif command == "dijkstra" and len(sys.argv) == 9:
            _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2, transport = sys.argv
            cli_dijkstra(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), transport)
        elif command == "astar" and len(sys.argv) == 9:
            _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2, transport = sys.argv
            cli_astar(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), transport)
        elif command == "astar_time" and len(sys.argv) == 9:
            _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2, transport = sys.argv
            cli_astar_time(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), transport)
        elif command == "vrp" and len(sys.argv) == 7:
            _, _, nodes_csv, edges_csv, input_txt, start_hhmm, transport = sys.argv
            cli_vrp(nodes_csv, edges_csv, input_txt, start_hhmm, transport)
        elif command == "node_to_xy_dist" and len(sys.argv) == 7:
            _, _, nodes_csv, edges_csv, x1, y1, node_id_str = sys.argv
            cli_node_to_xy_dist(nodes_csv, edges_csv, float(x1), float(y1), int(node_id_str))
        else:
            print(usage, end="")
    except Exception as e:
        print(f"Erro ao executar o comando '{command}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    execution_time = time.time() - start_time
    print("Tempo de execução total: %s segundos." % execution_time)