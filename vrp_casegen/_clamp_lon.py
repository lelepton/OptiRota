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
    