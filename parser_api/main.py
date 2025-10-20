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

import logging
import tempfile
from typing import List
from pathlib import Path
from _build_arg_parser import _build_arg_parser
from _parse_center import _parse_center
from download_osm_with_overpass import download_osm_with_overpass
from process_osm_to_csv import process_osm_to_csv

def main(argv: List[str] | None = None) -> int:
    '''
    Ponto de entrada da CLI: lê argumentos, baixa dados via Overpass e gera
    os CSVs de nós e arestas.
    
    Parâmetros
    ----------
    argv : List[str] | None  -> argumentos (para testes); None usa sys.argv[1:]

    Retorno
    -------
    int : código de saída (0 sucesso; >0 erro)
    
    Fluxo
    -----
    1) Monta parser e lê argumentos.
    2) Valida raio e faz parsing do centro.
    3) Define caminhos de saída e arquivo OSM (temp/persistente).
    4) Baixa OSM via Overpass.
    5) Converte OSM -> CSV (nodes/edges).
    6) Limpa arquivo temporário se aplicável.
    '''
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
