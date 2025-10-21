def _min_to_hhmm(minutes: int) -> str:
    '''
    Converte minutos desde 00:00 para "HH:MM" (zero-padded).
    '''

    minutes = max(0, min(24 * 60 - 1, minutes))
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
