import math
from .constantes import EARTH_RADIUS_M

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
