from classes_de_elementos.road_graph import RoadGraph
from funcoes_utilitarias.print_path_edges import print_path_edges

def cli_dijkstra(nodes_csv: str, edges_csv: str,
                 lat1: float, lon1: float, lat2: float, lon2: float,
                 highway: str) -> None:
    '''
    Calcula o caminho mínimo por Dijkstra entre dois pontos (lat/lon) e imprime
    a sequência de arestas com distâncias e tempo estimado por aresta e total.

    Parâmetros
    ----------
    nodes_csv         : str
    edges_csv         : str
    lat1, lon1        : float (origem)
    lat2, lon2        : float (destino)
    highway           : str   ('car' | 'bike' | 'foot')

    Retorno
    -------
    None
    '''

    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    source_node_id = graph.nearest_node(lat1, lon1, highway)
    path_edges, total_distance_m = graph.shortest_path_between(lat1, lon1, lat2, lon2, highway)
    if not path_edges:
        print("[]")
        return
    print_path_edges(path_edges, source_node_id, "Dijkstra")
    