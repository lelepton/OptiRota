import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Tuple, List, Dict

def iter_ways(osm_path: Path) -> Iterator[Tuple[List[int], Dict[str, str]]]:
    '''
    Itera sobre as ways do arquivo OSM e produz a lista ordenada de IDs de nós
    junto com o dicionário de tags. Não aplica filtro aqui.

    Parâmetros
    ----------
    osm_path : caminho do arquivo .osm de entrada

    Yield
    -----
    (node_ids, tags), onde:
      node_ids : List[int] com os IDs dos nós na ordem da polilinha
      tags     : Dict[str, str] com as tags da way (ex.: highway, name, oneway)

    Observações
    -----------
    - Usa iterparse e limpeza de elementos para controlar a memória.
    '''

    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "way":
            node_ids = [int(nd.attrib["ref"]) for nd in elem.findall("nd")]
            tags = {t.attrib["k"]: t.attrib.get("v", "") for t in elem.findall("tag")}
            yield node_ids, tags
            elem.clear()
