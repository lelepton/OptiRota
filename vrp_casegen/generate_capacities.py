import random
from typing import List

def generate_capacities(num_vehicles: int, fixed_caps: List[float] | None,
                        cap_min: float, cap_max: float) -> List[float]:
    '''
    Gera a lista de capacidades (em toneladas) para cada veículo. Se o
    usuário fornecer uma lista explícita, ela é usada; caso contrário, as
    capacidades são amostradas uniformemente em [cap_min, cap_max].

    Parâmetros
    ----------
    num_vehicles : int
    fixed_caps   : List[float] | None
    cap_min      : float
    cap_max      : float

    Retorno
    -------
    List[float] : capacidades (t) com tamanho = num_vehicles
    '''

    if fixed_caps:
        return list(map(float, fixed_caps[:num_vehicles]))
    return [round(random.uniform(cap_min, cap_max), 2) for _ in range(num_vehicles)]
