def _hhmm_to_min(hhmm: str) -> int:
    '''
    Converte "HH:MM" em minutos desde 00:00.
    '''

    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)
