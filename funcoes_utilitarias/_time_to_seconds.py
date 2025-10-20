def _time_to_seconds(time_hhmm: str) -> int:
    '''
    Converte "HH:MM" em segundos desde 00:00, para uso geral no módulo.

    Parâmetros
    ----------
    time_hhmm : str

    Retorno
    -------
    int : segundos no dia (0–86399)
    '''
    
    hh, mm = [int(x) for x in time_hhmm.strip().split(":")]
    return hh * 3600 + mm * 60