import math
import random
from typing import Tuple
from .constante import EARTH_RADIUS_M
from ._clamp_lon import _clamp_lon

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
