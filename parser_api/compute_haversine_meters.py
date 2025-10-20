import math
from constantes import EARTH_RADIUS_M

def compute_haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    '''
    Calcula a distância geodésica aproximada entre dois pontos (lat/lon) na
    superfície da Terra usando a fórmula de Haversine. Retorna a distância
    em METROS.

    Parâmetros
    ----------
    lat1, lon1 : float  -> coordenadas do primeiro ponto (graus decimais)
    lat2, lon2 : float  -> coordenadas do segundo ponto (graus decimais)

    Retorno
    -------
    float : distância aproximada em METROS entre os dois pontos

    Complexidade
    ------------
    O(1) — apenas operações aritméticas.
    '''

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2) ** 2))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
