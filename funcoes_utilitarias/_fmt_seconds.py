def _fmt_seconds(rem: float) -> str:
    '''
    Formata a parte de segundos remanescentes para impressão amigável:
    sem casas se inteiro; até 3 casas sem zeros à direita caso contrário.

    Parâmetros
    ----------
    rem : float

    Retorno
    -------
    str : representação amigável dos segundos
    '''

    return str(int(round(rem))) if abs(rem - round(rem)) < 1e-9 else f"{rem:.3f}".rstrip("0").rstrip(".")
