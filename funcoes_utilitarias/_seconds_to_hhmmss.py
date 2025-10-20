def _seconds_to_hhmmss(secs: float) -> str:
    '''
    Converte segundos do dia (float/int) no formato de string "HH:MM:SS".

    Parâmetros
    ----------
    secs : float

    Retorno
    -------
    str : horário formatado "HH:MM:SS"
    '''
    
    s = int(secs)
    hh = (s // 3600) % 24
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"
