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
import math
import random
from pathlib import Path
from typing import List, Tuple

# Constante
EARTH_RADIUS_M: float = 6_371_000.0  # raio médio da Terra em metros


def _clamp_lon(lon: float) -> float:
    '''
    Normaliza a longitude para o intervalo [-180, 180].

    Parâmetros
    ----------
    lon : float -> longitude em graus

    Retorno
    -------
    float : longitude normalizada
    '''

    while lon < -180.0:
        lon += 360.0
    while lon > 180.0:
        lon -= 360.0
    return lon


def random_point_in_circle(lat0: float, lon0: float, radius_m: float) -> Tuple[float, float]:
    '''
    Amostra um ponto (lat, lon) uniformemente por área dentro de um círculo
    de raio R (metros) em torno de (lat0, lon0).

    Parâmetros
    ----------
    lat0, lon0 : float -> centro em graus decimais
    radius_m   : float -> raio em METROS

    Retorno
    -------
    (float, float) : latitude, longitude do ponto gerado

    Observações
    -----------
    - Usa amostragem por r = R * sqrt(U), theta = 2πV, com correção de
      longitude por cos(latitude).
    '''

    # Amostragem por área
    u = random.random()
    v = random.random()
    r = radius_m * (u ** 0.5)
    theta = 2.0 * math.pi * v

    # Deslocamentos no plano local
    dx = r * math.cos(theta)
    dy = r * math.sin(theta)

    # Conversão aproximada: metros -> graus
    lat0_rad = math.radians(lat0)
    dlat = (dy / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (dx / (EARTH_RADIUS_M * math.cos(lat0_rad))) * (180.0 / math.pi)

    lat = lat0 + dlat
    lon = _clamp_lon(lon0 + dlon)
    return (lat, lon)


def _hhmm_to_min(hhmm: str) -> int:
    '''
    Converte "HH:MM" em minutos desde 00:00.
    '''

    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def _min_to_hhmm(minutes: int) -> str:
    '''
    Converte minutos desde 00:00 para "HH:MM" (zero-padded).
    '''

    minutes = max(0, min(24 * 60 - 1, minutes))
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def random_time_window(day_start_min: int, day_end_min: int, min_len_min: int, max_len_min: int) -> str:
    '''
    Gera uma janela de tempo "HH:MM-HH:MM" dentro do intervalo diário
    [day_start, day_end], ajustando para não ultrapassar limites.

    Parâmetros
    ----------
    day_start_min : int   -> início do dia em minutos (ex.: 8*60)
    day_end_min   : int   -> fim   do dia em minutos (ex.: 18*60)
    min_len_min   : int   -> duração mínima da janela em minutos
    max_len_min   : int   -> duração máxima da janela em minutos

    Retorno
    -------
    str : janela no formato "HH:MM-HH:MM"
    '''

    if min_len_min > max_len_min:
        min_len_min, max_len_min = max_len_min, min_len_min
    # Ponto de início possível: [start, end - min_len]
    latest_start = max(day_start_min, min(day_end_min - min_len_min, day_end_min))
    if latest_start <= day_start_min:
        start = day_start_min
    else:
        start = random.randint(day_start_min, latest_start)
    length = random.randint(min_len_min, max_len_min)
    end = min(start + length, day_end_min)
    if end <= start:
        end = min(start + min_len_min, day_end_min)
    return f"{_min_to_hhmm(start)}-{_min_to_hhmm(end)}"


def build_windows_string(multi_prob: float, day_start: str, day_end: str, min_len: int, max_len: int) -> str:
    '''
    Gera 1 ou 2 janelas de tempo com probabilidade `multi_prob`, concatenadas
    no formato esperado pelo VRP: "A-B, C-D".

    Parâmetros
    ----------
    multi_prob  : float -> probabilidade de gerar DUAS janelas
    day_start   : str   -> "HH:MM" início do dia
    day_end     : str   -> "HH:MM" fim do dia
    min_len     : int   -> duração mínima (min)
    max_len     : int   -> duração máxima (min)

    Retorno
    -------
    str : string das janelas
    '''

    ds = _hhmm_to_min(day_start)
    de = _hhmm_to_min(day_end)
    w1 = random_time_window(ds, de, min_len, max_len)
    if random.random() < multi_prob:
        w2 = random_time_window(ds, de, min_len, max_len)
        # Ordena por horário inicial
        a0 = _hhmm_to_min(w1.split("-")[0])
        b0 = _hhmm_to_min(w2.split("-")[0])
        return f"{w1}, {w2}" if a0 <= b0 else f"{w2}, {w1}"
    return w1


def generate_capacities(num_vehicles: int, fixed_caps: List[float] | None,
                        cap_min: float, cap_max: float) -> List[float]:
    '''
    Gera a lista de capacidades (em toneladas) para cada veículo. Se o
    usuário fornecer uma lista explícita, ela é usada; caso contrário, as
    capacidades são amostradas uniformemente em [cap_min, cap_max].

    Parâmetros
    ----------
    num_vehicles : int
    fixed_caps   : List[float] | None
    cap_min      : float
    cap_max      : float

    Retorno
    -------
    List[float] : capacidades (t) com tamanho = num_vehicles
    '''

    if fixed_caps:
        return list(map(float, fixed_caps[:num_vehicles]))
    return [round(random.uniform(cap_min, cap_max), 2) for _ in range(num_vehicles)]


def generate_weights(n_deliveries: int, w_min: float, w_max: float, max_cap: float | None) -> List[float]:
    '''
    Gera pesos (toneladas) para cada entrega. Se `max_cap` for informado,
    limita o peso máximo a `max_cap` para garantir viabilidade.

    Parâmetros
    ----------
    n_deliveries : int
    w_min        : float
    w_max        : float
    max_cap      : float | None

    Retorno
    -------
    List[float] : pesos das entregas
    '''

    hi = min(w_max, max_cap) if max_cap is not None else w_max
    lo = min(w_min, hi)
    return [round(random.uniform(lo, hi), 2) for _ in range(n_deliveries)]


def write_vrp_file(out_path: Path,
                   depot_lat: float, depot_lon: float,
                   capacities: List[float],
                   deliveries: List[Tuple[float, float, str, float]]) -> None:
    '''
    Escreve o arquivo final no formato aceito pelo VRP.

    Parâmetros
    ----------
    out_path     : Path
    depot_lat    : float
    depot_lon    : float
    capacities   : List[float]
    deliveries   : List[tuple(lat, lon, windows_str, weight_t)]

    Retorno
    -------
    None
    '''

    lines = []
    lines.append(f"{depot_lat:.6f} {depot_lon:.6f} {len(capacities)}")
    lines.append(" ".join(f"{c:.2f}" for c in capacities))
    for lat, lon, wstr, w in deliveries:
        lines.append(f"{lat:.6f} {lon:.6f} \"{wstr}\" {w:.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
