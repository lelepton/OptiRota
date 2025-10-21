from pathlib import Path
from typing import List, Tuple

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
