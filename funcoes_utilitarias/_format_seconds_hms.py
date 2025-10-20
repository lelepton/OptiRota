def _format_seconds_hms(total_seconds: float) -> str:
    '''
    Formata segundos em 'Hh MMmin SSs', 'Mmin SSs' ou 'Ss' conforme o caso.

    Parâmetros
    ----------
    total_seconds : float

    Retorno
    -------
    str : string amigável de duração
    '''
    
    seconds = int(round(total_seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}min {secs:02d}s"
    if minutes > 0:
        return f"{minutes}min {secs:02d}s"
    return f"{secs}s"
