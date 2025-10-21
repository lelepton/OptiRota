import random
from ._min_to_hhmm import _min_to_hhmm

def random_time_window(day_start_min: int, day_end_min: int, min_len_min: int, max_len_min: int) -> str:
    '''
    Gera uma janela de tempo "HH:MM-HH:MM" dentro do intervalo diário
    [day_start, day_end], ajustando para não ultrapassar limites.

    Parâmetros
    ----------
    day_start_min : int   -> início do dia em minutos (ex.: 8*60)
    day_end_min   : int   -> fim   do dia em minutos (ex.: 18*60)
    min_len_min   : int   -> duração mínima da janela em minutos
    max_len_min   : int   -> duração máxima da janela em minutos

    Retorno
    -------
    str : janela no formato "HH:MM-HH:MM"
    '''

    if min_len_min > max_len_min:
        min_len_min, max_len_min = max_len_min, min_len_min
    # Ponto de início possível: [start, end - min_len]
    latest_start = max(day_start_min, min(day_end_min - min_len_min, day_end_min))
    if latest_start <= day_start_min:
        start = day_start_min
    else:
        start = random.randint(day_start_min, latest_start)
    length = random.randint(min_len_min, max_len_min)
    end = min(start + length, day_end_min)
    if end <= start:
        end = min(start + min_len_min, day_end_min)
    return f"{_min_to_hhmm(start)}-{_min_to_hhmm(end)}"
