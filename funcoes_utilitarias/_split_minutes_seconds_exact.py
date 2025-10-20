def _split_minutes_seconds_exact(total_seconds: float) -> tuple[int, float]:
    '''
    Separa uma duração em segundos em (minutos inteiros, segundos remanescentes),
    sem arredondamentos externos, preservando exatidão.

    Parâmetros
    ----------
    total_seconds : float

    Retorno
    -------
    tuple[int, float] : (minutos_inteiros, segundos_restantes)
    '''

    mins = int(total_seconds // 60)
    rem = total_seconds - mins * 60
    return mins, rem
