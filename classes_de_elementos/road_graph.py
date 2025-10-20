import pandas as pd
import heapq
from typing import Dict, List, Tuple, Iterable, Optional
from classes_de_elementos.edge import Edge
from constantes.constantes import CAR_TAGS, BIKE_TAGS
from funcoes_utilitarias._coef_for_tag import _coef_for_tag
from funcoes_utilitarias._haversine_m import _haversine_m
from funcoes_utilitarias._is_edge_allowed import _is_edge_allowed
from funcoes_utilitarias._speed_mps_for_tag import _speed_mps_for_tag

class RoadGraph:
    '''
    Representa o grafo dirigido e ponderado G=(V,E) carregado a partir de CSVs.
    - nodes[osmid] = (lat, lon)
    - adj[u] = lista de objetos Edge saindo de u
    - w = coluna 'd' (metros); não recalculamos distância neste módulo.

    Observações
    -----------
    Cada aresta possui um único 'highway' e suas flags de acesso são definidas
    a partir dos conjuntos CAR_TAGS/BIKE_TAGS; allow_foot é sempre True.
    '''
    def __init__(self) -> None:
        '''
        Inicializa contêineres de nós e adjacência.
        '''
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.adj: Dict[int, List[Edge]] = {}

    def coef_for(self, tag: str) -> float:
        '''
        Obtém o coeficiente associado a uma tag de via.

        Parâmetros
        ----------
        tag : str (highway)

        Retorno
        -------
        float : coeficiente usado no peso artificial (A* com tempo)
        '''
        return _coef_for_tag(tag)

    def heuristic_distance_to_goal(self, node_id: int, target_node_id: int) -> float:
        '''
        Função reutilizável que retorna a distância Haversine (em metros) entre
        um nó "node_id" e o nó "target_node_id". Usada como heurística admissível para A*.
        '''
        node_lat, node_lon = self.nodes[node_id]
        target_lat, target_lon = self.nodes[target_node_id]
        return _haversine_m(node_lat, node_lon, target_lat, target_lon)
    
    def distance_from_node_to_point_m(self, node_id: int, lat: float, lon: float) -> float | None:
        '''
        Calcula a distância Haversine (em metros) entre um nó existente do grafo
        (identificado por 'node_id') e um ponto arbitrário (lat, lon).

        Parâmetros
        ----------
        node_id : int   (identificador do nó no grafo)
        lat     : float (latitude do ponto-alvo, em graus decimais)
        lon     : float (longitude do ponto-alvo, em graus decimais)

        Retorno
        -------
        float | None : distância em metros; None se o nó não existir no grafo
        '''
        if node_id not in self.nodes:
            return None
        node_lat, node_lon = self.nodes[node_id]
        return _haversine_m(node_lat, node_lon, float(lat), float(lon))

    @classmethod
    def load_graph(cls, nodes_csv: str, edges_csv: str) -> "RoadGraph":
        '''
        Lê os CSVs de nós (osmid,y,x) e arestas (u,v,d,highway[,name]) e constrói o grafo.

        Parâmetros
        ----------
        nodes_csv : caminho do CSV de nós
        edges_csv : caminho do CSV de arestas

        Retorno
        -------
        RoadGraph : instância pronta para consulta

        Observações
        -----------
        - Não recalcula distância; usa 'd' como peso.
        '''
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

    def neighbors(self, u: int, highway: Optional[str] = None) -> Iterable[Edge]:
        '''
        Retorna as arestas saindo de um nó u, opcionalmente filtradas por 'highway'.

        Parâmetros
        ----------
        u       : id do nó de origem
        highway : 'car' | 'bike' | 'foot' | None

        Retorno
        -------
        Iterable[Edge] : lista de arestas (filtrada quando 'highway' é fornecido)
        '''
        outgoing_edges = self.adj.get(u, [])
        if highway is None:
            return list(outgoing_edges)
        mode = highway.lower()
        if mode == "car":
            return [edge for edge in outgoing_edges if edge.allow_car]
        if mode == "bike":
            return [edge for edge in outgoing_edges if edge.allow_bike or edge.allow_car]
        return [edge for edge in outgoing_edges if edge.allow_foot or edge.allow_bike or edge.allow_car]  # foot

    def edge_count(self) -> int:
        '''
        Calcula a quantidade total de arestas do grafo.

        Retorno
        -------
        int : |E| (soma dos comprimentos das listas de adjacência)
        '''
        return sum(len(edge_list) for edge_list in self.adj.values())

    def node_count(self) -> int:
        '''
        Calcula a quantidade total de nós do grafo.

        Retorno
        -------
        int : |V| (tamanho de nodes)
        '''
        return len(self.nodes)

    def nearest_node(self, lat: float, lon: float, highway: str) -> int:
        '''
        Encontra o osmid do nó mais próximo às coordenadas (lat, lon), respeitando
        o filtro 'highway' (car | bike | foot).

        Parâmetros
        ----------
        lat     : latitude em graus decimais
        lon     : longitude em graus decimais
        highway : 'car' | 'bike' | 'foot'

        Retorno
        -------
        int : osmid do nó mais próximo entre os candidatos

        Observações
        -----------
        Inclusão por highway:
        - car  → allow_car
        - bike → allow_bike OU allow_car
        - foot → allow_foot OU allow_bike OU allow_car (todas)
        '''
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

    def _reconstruct_path(self, predecessor: dict[int, tuple[int, Edge]],
                          source_node_id: int, target_node_id: int) -> list[Edge]:
        '''
        Reconstrói a lista de arestas do caminho a partir de um dicionário de
        predecessores {nó: (nó_anterior, aresta_usada)}.
        '''
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

    def shortest_path_between(self, lat1: float, lon1: float,
                              lat2: float, lon2: float,
                              highway: str) -> tuple[list[Edge], float]:
        '''
        Executa Dijkstra entre dois pares (lat,lon), restringindo às arestas
        válidas para o 'highway' informado.

        Parâmetros
        ----------
        lat1, lon1 : coordenadas da origem
        lat2, lon2 : coordenadas do destino
        highway    : 'car' | 'bike' | 'foot'

        Retorno
        -------
        list[Edge] : lista de arestas na ordem origem→destino (vazia se não houver caminho)

        Observações
        -----------
        Passos: mapear coordenadas p/ nós com nearest_node; rodar Dijkstra; reconstruir a rota.
        '''

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

    def astar_shortest_path_between(self, lat1: float, lon1: float,
                                    lat2: float, lon2: float,
                                    highway: str) -> tuple[list[Edge], float, float]:
        '''
        Executa A* entre dois pares (lat,lon) usando a heurística de Haversine
        (distância em linha reta) até o destino, respeitando o filtro 'highway'.

        Parâmetros
        ----------
        lat1, lon1 : coordenadas da origem
        lat2, lon2 : coordenadas do destino
        highway    : 'car' | 'bike' | 'foot'

        Retorno
        -------
        list[Edge] : lista de arestas na ordem origem→destino (vazia se não houver caminho)

        Observações
        -----------
        - Heurística Haversine é admissível/consistente quando o custo é distância.
        - f(n) = g(n) + h(n), onde g(n) é a soma de pesos e h(n) é Haversine(n, destino).
        '''

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

    def astar_with_time_between(self, lat1: float, lon1: float,
                                lat2: float, lon2: float,
                                highway: str) -> tuple[list[Edge], float, float]:
        '''
        Executa A* com custo artificial (baseado em coeficientes por tag) e
        acumula o tempo real percorrido simultaneamente.

        Parâmetros
        ----------
        lat1, lon1 : coordenadas da origem
        lat2, lon2 : coordenadas do destino
        highway    : 'car' | 'bike' | 'foot'

        Retorno
        -------
        tuple[list[Edge], float, float] : (arestas, distância_total_m, tempo_total_s)

        Observações
        -----------
        - O "peso" minimizado é g = Σ (w / coef(tag)), enquanto o tempo real é somado à parte.
        - A heurística λ(n) = Haversine(n, destino) permanece sem escala/divisão.
        '''

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
    