
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
OSM -> CSV mínimo (VERSÃO COMENTADA)
----------------------------------------------------------------------------
Este script lê um arquivo OSM (.osm, formato XML do OpenStreetMap) e exporta:
  1) NÓS    : osmid, y (lat), x (lon)
  2) ARESTAS: u, v, d, name, highway

Arestas dirigidas
-----------------
- Cada par consecutivo de nós em uma 'way' vira 1 aresta. Se a 'way' NÃO for
  mão única ('oneway'), gravamos as duas direções: (u -> v) E (v -> u).
- Se 'oneway' for 'yes/true/1', gravamos apenas (u -> v).
- Se 'oneway' for '-1', gravamos apenas (v -> u).

Distância em METROS
-------------------
- 'd' é a distância Haversine (geodésica) entre os nós u e v, em METROS.
  Não há "peso" adimensional; é distância física.

Filtros básicos
---------------
- Mantemos 'highway' típicos de carro (drive), pedestres, ciclovias e 'path'.
- Ignoramos 'area=yes' (polígonos), 'highway=construction' e 'access=private'.

Dependências
------------
- Somente biblioteca padrão do Python (stdlib).
============================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

# Constantes globais
EARTH_RADIUS_M: float = 6_371_000.0

DRIVE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
PEDESTRIAN_SET = {"pedestrian", "footway", "steps"}
CYCLE_SET = {"cycleway"}


def compute_haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    '''
    Calcula a distância geodésica (grande círculo) entre dois pontos WGS84
    usando a fórmula de Haversine. O resultado é retornado em METROS.

    Parâmetros
    ----------
    lat1, lon1 : latitude e longitude do ponto A (graus decimais)
    lat2, lon2 : latitude e longitude do ponto B (graus decimais

    Retorno
    -------
    float : distância em metros ao longo da superfície da Terra

    Observações
    -----------
    - Converte ângulos para radianos antes de usar funções trigonométricas.
    - Usa o raio médio da Terra em metros (EARTH_RADIUS_M).
    '''

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2) ** 2))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


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


def is_relevant_way(tags: Dict[str, str]) -> bool:
    '''
    Define se uma way deve ser incluída no grafo exportado.
    Exclui áreas, trechos em construção e acesso privado.
    Inclui vias dirigíveis, vias de pedestres, ciclovias e 'path'.

    Parâmetros
    ----------
    tags : Dict[str, str] com as tags da way

    Retorno
    -------
    bool : True se a way for relevante; False caso contrário.
    '''

    highway = tags.get("highway")
    if highway is None:
        return False
    if tags.get("area") == "yes":
        return False
    if highway == "construction":
        return False
    if tags.get("access") == "private":
        return False
    return (
        (highway in DRIVE_HIGHWAYS)
        or (highway in PEDESTRIAN_SET)
        or (highway in CYCLE_SET)
        or (highway == "path")
    )


def dedupe_consecutive(ids_iter: Iterable[int]) -> List[int]:
    '''
    Retorna uma lista sem IDs duplicados consecutivos. Isso evita arestas de
    comprimento zero quando o dado OSM repete o mesmo nó em sequência.

    Parâmetros
    ----------
    ids_iter : iterável de IDs de nós

    Retorno
    -------
    List[int] : nova lista sem duplicatas consecutivas.
    '''
    result: List[int] = []
    last: int | None = None
    for nid in ids_iter:
        if nid != last:
            result.append(nid)
            last = nid
    return result


def write_nodes_csv(nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    '''
    Escreve o CSV de nós com colunas (osmid, y, x), onde y = latitude e
    x = longitude. A saída é ordenada pelo ID do nó para garantir determinismo.

    Parâmetros
    ----------
    nodes    : mapeamento osmid -> (lat, lon)
    csv_path : caminho do arquivo CSV de saída
    '''

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["osmid", "y", "x"])
        for osmid in sorted(nodes.keys()):
            lat, lon = nodes[osmid]
            writer.writerow([osmid, f"{lat:.10f}", f"{lon:.10f}"])


def write_edges_csv(osm_path: Path, nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    '''
    Lê as ways do OSM (filtrando irrelevantes) e escreve arestas dirigidas no
    CSV com colunas (u, v, d, name, highway). A distância é em metros (Haversine).
    A direcionalidade segue a tag 'oneway'; vias bidirecionais geram (u->v) e (v->u).

    Parâmetros
    ----------
    osm_path : caminho do arquivo .osm
    nodes    : mapeamento osmid -> (lat, lon)
    csv_path : caminho do arquivo CSV de saída
    '''

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "d", "name", "highway"])

        for node_ids, tags in iter_ways(osm_path):
            if not is_relevant_way(tags):
                continue

            name = tags.get("name", "") or ""
            highway = tags.get("highway", "") or ""
            oneway = (tags.get("oneway", "no") or "").lower()

            clean_ids = [nid for nid in dedupe_consecutive(node_ids) if nid in nodes]
            if len(clean_ids) < 2:
                continue

            def emit(a: int, b: int, dist_m: float) -> None:
                writer.writerow([a, b, f"{dist_m:.3f}", name, highway])

            for u, v in zip(clean_ids[:-1], clean_ids[1:]):
                lat_u, lon_u = nodes[u]
                lat_v, lon_v = nodes[v]
                d_m = compute_haversine_meters(lat_u, lon_u, lat_v, lon_v)

                if oneway in {"yes", "true", "1"}:
                    emit(u, v, d_m)
                elif oneway == "-1":
                    emit(v, u, d_m)
                else:
                    emit(u, v, d_m)
                    emit(v, u, d_m)


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


# CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write_osm_csv_refactored",
        description=(
            "Converte um arquivo OSM (.osm) em dois CSVs:\n"
            " - nodes: osmid,y,x\n"
            " - edges: u,v,d,name,highway (d em metros)"
        ),
    )
    parser.add_argument("--in", dest="osm_in", required=True, help="Caminho do arquivo .osm de entrada")
    parser.add_argument("--nodes", dest="nodes_csv", required=True, help="Caminho do CSV de nós (saída)")
    parser.add_argument("--edges", dest="edges_csv", required=True, help="Caminho do CSV de arestas (saída)")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Nível de log (ex.: INFO, DEBUG)")
    return parser

def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    osm_path = Path(args.osm_in).expanduser().resolve()
    nodes_csv = Path(args.nodes_csv).expanduser().resolve()
    edges_csv = Path(args.edges_csv).expanduser().resolve()

    if not osm_path.exists():
        logging.error("Arquivo de entrada não existe: %s", osm_path)
        return 1

    try:
        process_osm_to_csv(osm_path, nodes_csv, edges_csv)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Falha ao processar OSM: %s", exc)
        return 2

    logging.info("Concluído.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
