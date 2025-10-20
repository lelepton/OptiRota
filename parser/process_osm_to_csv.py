import logging
from pathlib import Path
from parser_nodes import parse_nodes
from write_nodes_csv import write_nodes_csv
from write_edges_csv import write_edges_csv

def process_osm_to_csv(osm_path: Path, nodes_csv: Path, edges_csv: Path) -> None:
    '''
    Pipeline de alto nível: coordena a leitura dos nós e a escrita dos dois
    CSVs (nós e arestas).

    Parâmetros
    ----------
    osm_path  : caminho do arquivo OSM de entrada
    nodes_csv : caminho do CSV de nós (saída)
    edges_csv : caminho do CSV de arestas (saída)
    '''

    logging.info("Lendo nós de %s", osm_path)
    nodes = parse_nodes(osm_path)

    logging.info("Gravando CSV de nós em %s", nodes_csv)
    write_nodes_csv(nodes, nodes_csv)

    logging.info("Gravando CSV de arestas em %s", edges_csv)
    write_edges_csv(osm_path, nodes, edges_csv)
