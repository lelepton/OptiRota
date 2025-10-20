import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

def parse_nodes(osm_path: Path) -> Dict[int, Tuple[float, float]]:
    '''
    Lê o arquivo OSM e retorna um dicionário que mapeia IDs de nós (osmid)
    para tuplas (latitude, longitude). Isso permite consulta O(1) das
    coordenadas quando formamos as arestas a partir das ways.

    Parâmetros
    ----------
    osm_path : caminho do arquivo .osm de entrada

    Retorno
    -------
    Dict[int, Tuple[float, float]] : mapeamento osmid -> (lat, lon)

    Observações
    -----------
    - Usa iterparse no evento "end" e limpa os elementos para reduzir uso de memória.
    '''

    nodes: Dict[int, Tuple[float, float]] = {}
    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "node":
            osmid = int(elem.attrib["id"])
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            nodes[osmid] = (lat, lon)
            elem.clear()
    return nodes
