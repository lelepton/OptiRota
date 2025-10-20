from classes_de_elementos.road_graph import RoadGraph

def cli_nearest(nodes_csv: str, edges_csv: str, lat: float, lon: float, highway: str) -> None:
    '''
    Encontra o nó roteável mais próximo às coordenadas informadas, respeitando o
    filtro de modal (car|bike|foot), e imprime o osmid.

    Parâmetros
    ----------
    nodes_csv : str
    edges_csv : str
    lat, lon  : float
    highway   : str ('car' | 'bike' | 'foot')

    Retorno
    -------
    None
    '''

    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    print(graph.nearest_node(lat, lon, highway))
    