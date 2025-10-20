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

import logging
from pathlib import Path
from typing import List
from _build_arg_parser import _build_arg_parser
from process_osm_to_csv import process_osm_to_csv

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
