import logging
from pathlib import Path
from .parse_nodes import parse_nodes
from .write_nodes_csv import write_nodes_csv
from .write_edges_csv import write_edges_csv

def process_osm_to_csv(osm_path: Path, nodes_csv: Path, edges_csv: Path) -> None:
    '''
    Pipeline de conversão: carrega nós (nodes.csv) e em seguida extrai as
    arestas (edges.csv) a partir das ways relevantes.
    
    Parâmetros
    ----------
    osm_path  : Path  -> caminho do arquivo .osm (XML)
    nodes_csv : Path  -> saída do CSV de nós
    edges_csv : Path  -> saída do CSV de arestas
    
    Retorno
    -------
    None
    
    Complexidade
    ------------
    Dominada por parse_nodes e write_edges_csv.
    '''

    logging.info("Lendo nós de %s", osm_path)
    nodes = parse_nodes(osm_path)

    logging.info("Gravando CSV de nós em %s", nodes_csv)
    write_nodes_csv(nodes, nodes_csv)

    logging.info("Gravando CSV de arestas em %s", edges_csv)
    write_edges_csv(osm_path, nodes, edges_csv)
