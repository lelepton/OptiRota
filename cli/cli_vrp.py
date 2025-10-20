from classes_de_elementos.road_graph import RoadGraph
from funcoes_utilitarias.parse_vrp_input_file import parse_vrp_input_file
from algoritmo_do_vrp.vrp_heuristic import VRP_heuristic

def cli_vrp(nodes_csv: str, edges_csv: str,
            input_txt: str, start_HH_MM: str, highway: str = "car") -> None:
    '''
    Executa a heurística VRP a partir de um arquivo de entrada em texto,
    carregando o grafo a partir dos CSVs e repassando os parâmetros.

    Parâmetros
    ----------
    nodes_csv  : str (caminho para CSV de nós)
    edges_csv  : str (caminho para CSV de arestas)
    input_txt  : str (arquivo de entrada com origem/capacidade e entregas)
    start_HH_MM: str (horário de partida "HH:MM")
    highway    : str ('car' | 'bike' | 'foot')

    Retorno
    -------
    None (imprime o plano de entregas)
    '''

    graph = RoadGraph.load_graph(nodes_csv, edges_csv)
    origin_lat, origin_lon, capacities_tons_L, deliveries_raw = parse_vrp_input_file(input_txt)
    VRP_heuristic(graph, origin_lat, origin_lon, start_HH_MM, capacities_tons_L, deliveries_raw, highway)
    