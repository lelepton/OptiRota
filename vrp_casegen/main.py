#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
Gerador de casos de teste para VRP (compatível com optirota.py)
----------------------------------------------------------------------------
Gera um arquivo de entrada no formato esperado pelo VRP modificado:
  1) <depot_lat> <depot_lon> <num_veiculos>
  2) <cap1> <cap2> ... <capN>
  3+) <lat> <lon> "HH:MM-HH:MM[, HH:MM-HH:MM]" <peso_ton>

Os destinos são amostrados em um círculo de raio R (em METROS) ao redor
do centro informado. As janelas de tempo são geradas dentro de um
intervalo diário (ex.: 08:00–18:00), com duração aleatória.
As capacidades e pesos podem ser fixados ou gerados em faixas.

Sem dependências externas (apenas stdlib).
============================================================================
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple
from .generate_capacities import generate_capacities
from .generate_weights import generate_weights
from .random_point_in_circle import random_point_in_circle
from .build_windows_string import build_windows_string
from .write_vrp_file import write_vrp_file

def main(argv: list[str] | None = None) -> int:
    '''
    Ponto de entrada da CLI do gerador.
    '''

    parser = argparse.ArgumentParser(
        prog="vrp_casegen",
        description=(
            "Gera arquivo de teste para VRP a partir de centro, raio, "
            "número de veículos e número de destinos."
        ),
    )
    parser.add_argument("--center", required=True, help="Centro 'lat,lon' (graus decimais)")
    parser.add_argument("--radius", type=float, required=True, help="Raio em METROS")
    parser.add_argument("--vehicles", type=int, required=True, help="Número de veículos de entrega (N)")
    parser.add_argument("--destinations", type=int, required=True, help="Número de coordenadas de destino (M)")
    parser.add_argument("--out", required=True, help="Arquivo de saída (texto)")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade")

    # Capacidades e pesos (opcionais)
    parser.add_argument("--caps", type=str, default=None,
                        help="Capacidades explícitas: ex. '3,4.5,5'. Se ausente, usa faixas.")
    parser.add_argument("--cap-min", type=float, default=3.0, help="Capacidade mínima por veículo (t)")
    parser.add_argument("--cap-max", type=float, default=5.0, help="Capacidade máxima por veículo (t)")
    parser.add_argument("--w-min", type=float, default=0.3, help="Peso mínimo por entrega (t)")
    parser.add_argument("--w-max", type=float, default=1.5, help="Peso máximo por entrega (t)")

    # Janelas de tempo
    parser.add_argument("--day-start", type=str, default="08:00", help="Início do dia (HH:MM)")
    parser.add_argument("--day-end", type=str, default="18:00", help="Fim do dia (HH:MM)")
    parser.add_argument("--win-min", type=int, default=45, help="Duração mínima da janela (min)")
    parser.add_argument("--win-max", type=int, default=120, help="Duração máxima da janela (min)")
    parser.add_argument("--multi-prob", type=float, default=0.25, help="Probabilidade de duas janelas por entrega [0..1]")

    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(int(args.seed))

    try:
        lat_str, lon_str = [s.strip() for s in args.center.split(",", 1)]
        lat0 = float(lat_str); lon0 = float(lon_str)
    except Exception as exc:
        raise SystemExit(f"--center inválido: {args.center!r}") from exc

    if args.vehicles <= 0 or args.destinations <= 0 or args.radius <= 0:
        raise SystemExit("Parâmetros inválidos: --vehicles, --destinations e --radius devem ser positivos.")

    # Capacidades
    fixed_caps = None
    if args.caps:
        fixed_caps = [float(tok) for tok in args.caps.replace(",", " ").split() if tok.strip()]
    capacities = generate_capacities(args.vehicles, fixed_caps, args.cap_min, args.cap_max)

    # Pesos
    max_cap = max(capacities) if capacities else None
    weights = generate_weights(args.destinations, args.w_min, args.w_max, max_cap)

    # Destinos
    deliveries: List[Tuple[float, float, str, float]] = []
    for i in range(args.destinations):
        lat, lon = random_point_in_circle(lat0, lon0, args.radius)
        wstr = build_windows_string(args.multi_prob, args.day_start, args.day_end, args.win_min, args.win_max)
        deliveries.append((lat, lon, wstr, weights[i]))

    # Escrever arquivo
    out_path = Path(args.out).expanduser().resolve()
    write_vrp_file(out_path, lat0, lon0, capacities, deliveries)
    print(f"Arquivo gerado: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
