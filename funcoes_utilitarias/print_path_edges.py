from funcoes_utilitarias._format_seconds_hms import _format_seconds_hms
from funcoes_utilitarias._speed_mps_for_tag import _speed_mps_for_tag

def print_path_edges(path_edges, source_node_id: int, label_total: str) -> None:
    '''
    Imprime a sequência de arestas com distância por aresta, distância
    cumulativa até então e metadados, além da distância e tempo totais ao final

    Parâmetros
    ----------
    path_edges : list[Edge] (na ordem origem→destino)
    source_node_id : int (id do nó de origem para imprimir "u -> v")
    label_total : str (rótulo do total, ex.: "Dijkstra" ou "A*")
    '''

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
    