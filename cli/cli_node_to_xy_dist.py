from classes_de_elementos.road_graph import RoadGraph

def cli_node_to_xy_dist(nodes_csv: str, edges_csv: str, x1: float, y1: float, node_id: int) -> None:
    '''
    Carrega o grafo e imprime a distância (em metros) do nó <node_id> ao ponto
    (x1, y1). Convenção: x = longitude, y = latitude.

    Parâmetros
    ----------
    nodes_csv : str (caminho para CSV de nós)
    edges_csv : str (caminho para CSV de arestas)
    x1        : float (longitude do ponto)
    y1        : float (latitude do ponto)
    node_id   : int   (identificador do nó no grafo)

    Retorno
    -------
    None (imprime a distância em metros ou a mensagem de ausência do nó)
    '''

    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    if node_id not in graph.nodes:
        print("Não está presente nos arquivos de entrada")
        return
    distance_m = graph.distance_from_node_to_point_m(node_id, float(y1), float(x1))
    print(f"{distance_m:.3f}")
    