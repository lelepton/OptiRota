#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
OSM -> CSV a partir de (lat, lon, R) via Overpass API
----------------------------------------------------------------------------
Baixa dados do OpenStreetMap usando a Overpass API para um círculo
definido por centro (latitude, longitude) e raio em METROS, e gera dois CSVs:

  1) nodes.csv : osmid, y (lat), x (lon)
  2) edges.csv : u, v, d, name, highway   (arestas dirigidas; d em METROS)

Filtros (iguais ao parser anterior):
- Inclui vias dirigíveis (carros), pedestres, ciclovias e 'path'.
- Ignora 'area=yes', 'highway=construction' e 'access=private'.

Dependências: somente stdlib.
============================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

# -------------------------------------------------------------------------
# Constantes globais
# -------------------------------------------------------------------------
# Raio médio da Terra em metros, usado na distância de Haversine.
EARTH_RADIUS_M: float = 6_371_000.0

# Conjunto de valores de 'highway' considerados dirigíveis.
DRIVE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}

# Conjuntos específicos para pedestres e ciclovias.
PEDESTRIAN_SET = {"pedestrian", "footway", "steps"}
CYCLE_SET = {"cycleway"}

# Endpoint padrão da Overpass API (pode ser trocado por espelhos).
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# -------------------------------------------------------------------------
# Função: compute_haversine_meters
# -------------------------------------------------------------------------
# Descrição
# ---------
# Calcula a distância geodésica aproximada entre dois pontos (lat/lon) na
# superfície da Terra usando a fórmula de Haversine. Retorna a distância
# em METROS.
#
# Parâmetros
# ----------
# lat1, lon1 : float  -> coordenadas do primeiro ponto (graus decimais)
# lat2, lon2 : float  -> coordenadas do segundo ponto (graus decimais)
#
# Retorno
# -------
# float : distância aproximada em METROS entre os dois pontos
#
# Complexidade
# ------------
# O(1) — apenas operações aritméticas.
def compute_haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2) ** 2))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# -------------------------------------------------------------------------
# Função: parse_nodes
# -------------------------------------------------------------------------
# Descrição
# ---------
# Lê um arquivo OSM (XML) e retorna um dicionário {osmid: (lat, lon)} com
# todos os nós encontrados.
#
# Parâmetros
# ----------
# osm_path : Path  -> caminho do arquivo .osm (XML)
#
# Retorno
# -------
# Dict[int, Tuple[float, float]] : mapeamento de nós
#
# Complexidade
# ------------
# O(N) no número de elementos <node/> do arquivo.
#
# Observações
# -----------
# - Usa iterparse com evento "end" e limpa elementos para reduzir memória.
def parse_nodes(osm_path: Path) -> Dict[int, Tuple[float, float]]:
    nodes: Dict[int, Tuple[float, float]] = {}
    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "node":
            osmid = int(elem.attrib["id"])
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            nodes[osmid] = (lat, lon)
            elem.clear()
    return nodes


# -------------------------------------------------------------------------
# Função: iter_ways
# -------------------------------------------------------------------------
# Descrição
# ---------
# Itera sobre as ways do arquivo OSM e produz a lista ordenada de IDs de nós
# junto com o dicionário de tags. Não aplica filtro aqui.
#
# Parâmetros
# ----------
# osm_path : Path  -> caminho do arquivo .osm de entrada
#
# Yield
# -----
# (node_ids, tags), onde:
#   node_ids : List[int]          -> IDs dos nós na ordem da polilinha
#   tags     : Dict[str, str]     -> tags da way (ex.: highway, name, oneway)
#
# Complexidade
# ------------
# O(W + T), onde W é o número de ways e T o total de tags das ways.
#
# Observações
# -----------
# - Usa iterparse e limpeza de elementos para controlar a memória.
def iter_ways(osm_path: Path) -> Iterator[Tuple[List[int], Dict[str, str]]]:
    for _, elem in ET.iterparse(str(osm_path), events=("end",)):
        if elem.tag == "way":
            node_ids = [int(nd.attrib["ref"]) for nd in elem.findall("nd")]
            tags = {t.attrib["k"]: t.attrib.get("v", "") for t in elem.findall("tag")}
            yield node_ids, tags
            elem.clear()


# -------------------------------------------------------------------------
# Função: is_relevant_way
# -------------------------------------------------------------------------
# Descrição
# ---------
# Aplica os filtros de relevância para inclusão de ways no grafo viário.
# Mantém vias dirigíveis, pedestres, ciclovias e 'path'; descarta áreas,
# vias em construção e acesso privado.
#
# Parâmetros
# ----------
# tags : Dict[str, str]  -> tags da way (ex.: {'highway': 'residential', ...})
#
# Retorno
# -------
# bool : True se a way deve ser incluída; False caso contrário.
#
# Complexidade
# ------------
# O(1) — consultas de dicionário e conjuntos.
#
# Regras
# ------
# - Exige presença de 'highway'.
# - Exclui area=yes, highway=construction, access=private.
# - Inclui se highway ∈ DRIVE_HIGHWAYS ∪ PEDESTRIAN_SET ∪ CYCLE_SET ∪ {'path'}.
def is_relevant_way(tags: Dict[str, str]) -> bool:
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


# -------------------------------------------------------------------------
# Função: dedupe_consecutive
# -------------------------------------------------------------------------
# Descrição
# ---------
# Remove duplicatas consecutivas de uma sequência de IDs, preservando a ordem.
# Útil para polilinhas com nós repetidos adjacentes.
#
# Parâmetros
# ----------
# ids_iter : Iterable[int]  -> sequência de IDs
#
# Retorno
# -------
# List[int] : sequência sem repetições consecutivas
#
# Complexidade
# ------------
# O(N), onde N é o comprimento da sequência.
def dedupe_consecutive(ids_iter: Iterable[int]) -> List[int]:
    result: List[int] = []
    last: int | None = None
    for nid in ids_iter:
        if nid != last:
            result.append(nid)
            last = nid
    return result


# -------------------------------------------------------------------------
# Função: write_nodes_csv
# -------------------------------------------------------------------------
# Descrição
# ---------
# Gera o arquivo nodes.csv com cabeçalho (osmid, y, x), onde y=lat e x=lon.
#
# Parâmetros
# ----------
# nodes    : Dict[int, Tuple[float, float]]  -> {osmid: (lat, lon)}
# csv_path : Path                            -> caminho do CSV de saída
#
# Retorno
# -------
# None
#
# Complexidade
# ------------
# O(N log N) devido à ordenação dos IDs para saída determinística.
def write_nodes_csv(nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["osmid", "y", "x"])
        for osmid in sorted(nodes.keys()):
            lat, lon = nodes[osmid]
            writer.writerow([osmid, f"{lat:.10f}", f"{lon:.10f}"])


# -------------------------------------------------------------------------
# Função: write_edges_csv
# -------------------------------------------------------------------------
# Descrição
# ---------
# Gera o arquivo edges.csv com cabeçalho (u, v, d, name, highway). Para cada
# par consecutivo de nós (u, v) em uma way relevante, escreve arestas em
# conformidade com 'oneway'. O campo d é a distância Haversine em METROS.
#
# Parâmetros
# ----------
# osm_path : Path                             -> caminho do arquivo .osm (XML)
# nodes    : Dict[int, Tuple[float, float]]   -> {osmid: (lat, lon)}
# csv_path : Path                             -> caminho do CSV de arestas
#
# Retorno
# -------
# None
#
# Complexidade
# ------------
# O(W * K), onde W é o número de ways relevantes e K o número médio de
# segmentos (pares consecutivos de nós) por way.
#
# Observações
# -----------
# - Aplica dedupe_consecutive e ignora nós ausentes na tabela de nós.
# - oneway in {'yes','true','1'} -> (u→v)
# - oneway == '-1'               -> (v→u)
# - caso contrário               -> (u→v) e (v→u)
def write_edges_csv(osm_path: Path, nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
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


# -------------------------------------------------------------------------
# Função: process_osm_to_csv
# -------------------------------------------------------------------------
# Descrição
# ---------
# Pipeline de conversão: carrega nós (nodes.csv) e em seguida extrai as
# arestas (edges.csv) a partir das ways relevantes.
#
# Parâmetros
# ----------
# osm_path  : Path  -> caminho do arquivo .osm (XML)
# nodes_csv : Path  -> saída do CSV de nós
# edges_csv : Path  -> saída do CSV de arestas
#
# Retorno
# -------
# None
#
# Complexidade
# ------------
# Dominada por parse_nodes e write_edges_csv.
def process_osm_to_csv(osm_path: Path, nodes_csv: Path, edges_csv: Path) -> None:
    logging.info("Lendo nós de %s", osm_path)
    nodes = parse_nodes(osm_path)

    logging.info("Gravando CSV de nós em %s", nodes_csv)
    write_nodes_csv(nodes, nodes_csv)

    logging.info("Gravando CSV de arestas em %s", edges_csv)
    write_edges_csv(osm_path, nodes, edges_csv)


# -------------------------------------------------------------------------
# Função: build_overpass_query
# -------------------------------------------------------------------------
# Descrição
# ---------
# Monta a consulta Overpass QL (saída XML) que seleciona ways com 'highway'
# dentro de um círculo definido por (lat, lon) e raio (em METROS), e usa a
# recursão '>' para incluir os nós.
#
# Parâmetros
# ----------
# lat, lon  : float  -> centro do círculo (graus decimais)
# radius_m  : float  -> raio em METROS
#
# Retorno
# -------
# str : consulta Overpass QL formatada
#
# Complexidade
# ------------
# O(1) — apenas construção de string.
#
# Observações
# -----------
# - Usa '[out:xml]' para aproveitar o parser XML existente.
# - Timeout ampliado (180s) para áreas densas.
def build_overpass_query(lat: float, lon: float, radius_m: float) -> str:
    q = f"""
[out:xml][timeout:180];
(
  way[highway](around:{int(radius_m)},{lat:.7f},{lon:.7f});
);
(._;>;);
out body;
"""
    return "\n".join(line.strip() for line in q.strip().splitlines())


# -------------------------------------------------------------------------
# Função: download_osm_with_overpass
# -------------------------------------------------------------------------
# Descrição
# ---------
# Executa POST na Overpass API com a consulta gerada e salva o XML em
# dest_osm. Implementa retry com backoff simples para falhas transitórias.
#
# Parâmetros
# ----------
# lat, lon     : float  -> centro do círculo (graus decimais)
# radius_m     : float  -> raio em METROS
# dest_osm     : Path   -> caminho do arquivo .osm a salvar
# overpass_url : str    -> endpoint da Overpass (opcional)
# retries      : int    -> tentativas em caso de falha (padrão: 3)
# backoff_sec  : float  -> fator de espera entre tentativas (padrão: 2.5)
#
# Retorno
# -------
# None
#
# Complexidade
# ------------
# O(Tamanho_da_resposta) para E/S; latência de rede variável.
#
# Observações
# -----------
# - Define User-Agent explícito e Content-Type padrão de formulários.
# - Em falha, registra warnings e levanta exceção após esgotar tentativas.
def download_osm_with_overpass(
    lat: float,
    lon: float,
    radius_m: float,
    dest_osm: Path,
    overpass_url: str = DEFAULT_OVERPASS_URL,
    retries: int = 3,
    backoff_sec: float = 2.5,
) -> None:
    dest_osm.parent.mkdir(parents=True, exist_ok=True)
    query = build_overpass_query(lat, lon, radius_m)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    req = urllib.request.Request(
        overpass_url,
        data=data,
        method="POST",
        headers={
            "User-Agent": "osm-csv-downloader/1.0 (+https://example.org)",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Overpass HTTP {resp.status}")
                content = resp.read()
                dest_osm.write_bytes(content)
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logging.warning("Falha ao baixar (tentativa %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
            else:
                break

    raise RuntimeError(f"Não foi possível baixar dados da Overpass: {last_err}")


# -------------------------------------------------------------------------
# Função: _build_arg_parser
# -------------------------------------------------------------------------
# Descrição
# ---------
# Constrói e retorna o ArgumentParser da CLI.
#
# Parâmetros
# ----------
# (nenhum)
#
# Retorno
# -------
# argparse.ArgumentParser : parser configurado
#
# Opções expostas
# ---------------
# --center "lat,lon"  | --radius METROS | --nodes PATH | --edges PATH
# --overpass-url URL  | --save-osm PATH | --log-level LEVEL
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osm_circle_to_csv",
        description=(
            "Baixa OSM via Overpass para um círculo (lat,lon,R em metros) e gera dois CSVs:\n"
            " - nodes: osmid,y,x\n"
            " - edges: u,v,d,name,highway (d em metros)"
        ),
    )
    parser.add_argument(
        "--center",
        required=True,
        help="Coordenadas no formato 'lat,lon' (graus decimais). Ex.: -9.648123,-35.717456",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Raio em METROS para a busca circular (around). Ex.: 2000",
    )
    parser.add_argument("--nodes", dest="nodes_csv", required=True, help="Caminho do CSV de nós (saída)")
    parser.add_argument("--edges", dest="edges_csv", required=True, help="Caminho do CSV de arestas (saída)")
    parser.add_argument(
        "--overpass-url",
        default=DEFAULT_OVERPASS_URL,
        help=f"URL do endpoint Overpass (padrão: {DEFAULT_OVERPASS_URL})",
    )
    parser.add_argument(
        "--save-osm",
        dest="save_osm",
        default=None,
        help="Se fornecido, salva o .osm baixado neste caminho (útil para cache/inspeção).",
    )
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Nível de log (ex.: INFO, DEBUG)")
    return parser


# -------------------------------------------------------------------------
# Função: _parse_center
# -------------------------------------------------------------------------
# Descrição
# ---------
# Faz o parsing do parâmetro --center no formato 'lat,lon', valida intervalos
# e retorna a tupla (lat, lon) como floats.
#
# Parâmetros
# ----------
# center_str : str  -> string no formato 'lat,lon' (graus decimais)
#
# Retorno
# -------
# Tuple[float, float] : (lat, lon)
#
# Complexidade
# ------------
# O(1)
#
# Observações
# -----------
# - Lança argparse.ArgumentTypeError em caso de formato inválido ou limites.
def _parse_center(center_str: str) -> Tuple[float, float]:
    try:
        lat_str, lon_str = [s.strip() for s in center_str.split(",", 1)]
        lat = float(lat_str)
        lon = float(lon_str)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Formato inválido para --center: {center_str!r}") from exc
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise argparse.ArgumentTypeError(f"Coordenadas fora do intervalo válido: lat={lat}, lon={lon}")
    return lat, lon


# -------------------------------------------------------------------------
# Função: main
# -------------------------------------------------------------------------
# Descrição
# ---------
# Ponto de entrada da CLI: lê argumentos, baixa dados via Overpass e gera
# os CSVs de nós e arestas.
#
# Parâmetros
# ----------
# argv : List[str] | None  -> argumentos (para testes); None usa sys.argv[1:]
#
# Retorno
# -------
# int : código de saída (0 sucesso; >0 erro)
#
# Fluxo
# -----
# 1) Monta parser e lê argumentos.
# 2) Valida raio e faz parsing do centro.
# 3) Define caminhos de saída e arquivo OSM (temp/persistente).
# 4) Baixa OSM via Overpass.
# 5) Converte OSM -> CSV (nodes/edges).
# 6) Limpa arquivo temporário se aplicável.
def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    lat, lon = _parse_center(args.center)
    if args.radius <= 0:
        logging.error("O raio deve ser positivo (em metros). Valor recebido: %s", args.radius)
        return 2

    nodes_csv = Path(args.nodes_csv).expanduser().resolve()
    edges_csv = Path(args.edges_csv).expanduser().resolve()

    if args.save_osm:
        osm_path = Path(args.save_osm).expanduser().resolve()
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="overpass_", suffix=".osm", delete=False)
        osm_path = Path(tmp.name)
        tmp.close()

    try:
        logging.info("Baixando OSM (lat=%.6f, lon=%.6f, R=%.1fm) via %s", lat, lon, args.radius, args.overpass_url)
        download_osm_with_overpass(lat, lon, args.radius, osm_path, args.overpass_url)

        process_osm_to_csv(osm_path, nodes_csv, edges_csv)

    except Exception as exc:  # noqa: BLE001
        logging.exception("Falha no processamento: %s", exc)
        return 3
    finally:
        if not args.save_osm and osm_path.exists():
            try:
                osm_path.unlink()
            except Exception:
                pass

    logging.info("Concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
