import random
from ._hhmm_to_min import _hhmm_to_min
from .random_time_window import random_time_window

def build_windows_string(multi_prob: float, day_start: str, day_end: str, min_len: int, max_len: int) -> str:
    '''
    Gera 1 ou 2 janelas de tempo com probabilidade `multi_prob`, concatenadas
    no formato esperado pelo VRP: "A-B, C-D".

    Parâmetros
    ----------
    multi_prob  : float -> probabilidade de gerar DUAS janelas
    day_start   : str   -> "HH:MM" início do dia
    day_end     : str   -> "HH:MM" fim do dia
    min_len     : int   -> duração mínima (min)
    max_len     : int   -> duração máxima (min)

    Retorno
    -------
    str : string das janelas
    '''

    ds = _hhmm_to_min(day_start)
    de = _hhmm_to_min(day_end)
    w1 = random_time_window(ds, de, min_len, max_len)
    if random.random() < multi_prob:
        w2 = random_time_window(ds, de, min_len, max_len)
        # Ordena por horário inicial
        a0 = _hhmm_to_min(w1.split("-")[0])
        b0 = _hhmm_to_min(w2.split("-")[0])
        return f"{w1}, {w2}" if a0 <= b0 else f"{w2}, {w1}"
    return w1
