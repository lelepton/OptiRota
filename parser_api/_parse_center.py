import argparse
from typing import Tuple

def _parse_center(center_str: str) -> Tuple[float, float]:
    '''
    Faz o parsing do parâmetro --center no formato 'lat,lon', valida intervalos
    e retorna a tupla (lat, lon) como floats.

    Parâmetros
    ----------
    center_str : str  -> string no formato 'lat,lon' (graus decimais)

    Retorno
    -------
    Tuple[float, float] : (lat, lon)

    Complexidade
    ------------
    O(1)

    Observações
    -----------
    - Lança argparse.ArgumentTypeError em caso de formato inválido ou limites.
    '''

    try:
        lat_str, lon_str = [s.strip() for s in center_str.split(",", 1)]
        lat = float(lat_str)
        lon = float(lon_str)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Formato inválido para --center: {center_str!r}") from exc
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise argparse.ArgumentTypeError(f"Coordenadas fora do intervalo válido: lat={lat}, lon={lon}")
    return lat, lon
