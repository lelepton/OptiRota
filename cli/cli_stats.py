from classes_de_elementos.road_graph import RoadGraph

def cli_stats(nodes_csv: str, edges_csv: str) -> None:
    '''
    Carrega o grafo a partir dos CSVs e imprime estatísticas básicas: |V| e |E|.
    
    Parâmetros
    ----------
    nodes_csv : str (caminho para CSV de nós)
    edges_csv : str (caminho para CSV de arestas)

    Retorno
    -------
    None
    '''
    
    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    print(f"RoadGraph |V|={graph.node_count()} |E|={graph.edge_count()}")
