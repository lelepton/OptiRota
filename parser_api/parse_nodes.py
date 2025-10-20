import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

def parse_nodes(osm_path: Path) -> Dict[int, Tuple[float, float]]:
    '''
    Lê um arquivo OSM (XML) e retorna um dicionário {osmid: (lat, lon)} com
    todos os nós encontrados.

    Parâmetros
    ----------
    osm_path : Path  -> caminho do arquivo .osm (XML)

    Retorno
    -------
    Dict[int, Tuple[float, float]] : mapeamento de nós

    Complexidade
    ------------
    O(N) no número de elementos <node/> do arquivo.

    Observações
    -----------
    - Usa iterparse com evento "end" e limpa elementos para reduzir memória.
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
