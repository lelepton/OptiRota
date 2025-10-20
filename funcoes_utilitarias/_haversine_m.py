import math

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    '''
    Calcula a distância Haversine entre dois pontos (lat, lon) em metros.
    
    Parâmetros
    ----------
    lat1, lon1 : float (ponto A)
    lat2, lon2 : float (ponto B)

    Retorno
    ----------
    float : distância em metros
    '''
    
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * 6371008.8 * math.asin(math.sqrt(a))
