import random
from typing import List

def generate_weights(n_deliveries: int, w_min: float, w_max: float, max_cap: float | None) -> List[float]:
    '''
    Gera pesos (toneladas) para cada entrega. Se `max_cap` for informado,
    limita o peso máximo a `max_cap` para garantir viabilidade.

    Parâmetros
    ----------
    n_deliveries : int
    w_min        : float
    w_max        : float
    max_cap      : float | None

    Retorno
    -------
    List[float] : pesos das entregas
    '''

    hi = min(w_max, max_cap) if max_cap is not None else w_max
    lo = min(w_min, hi)
    return [round(random.uniform(lo, hi), 2) for _ in range(n_deliveries)]
