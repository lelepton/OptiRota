from typing import Dict
from constantes import DRIVE_HIGHWAYS, PEDESTRIAN_SET, CYCLE_SET

def is_relevant_way(tags: Dict[str, str]) -> bool:
    '''
    Aplica os filtros de relevância para inclusão de ways no grafo viário.
    Mantém vias dirigíveis, pedestres, ciclovias e 'path'; descarta áreas,
    vias em construção e acesso privado.

    Parâmetros
    ----------
    tags : Dict[str, str]  -> tags da way (ex.: {'highway': 'residential', ...})

    Retorno
    -------
    bool : True se a way deve ser incluída; False caso contrário.

    Complexidade
    ------------
    O(1) — consultas de dicionário e conjuntos.

    Regras
    ------
    - Exige presença de 'highway'.
    - Exclui area=yes, highway=construction, access=private.
    - Inclui se highway ∈ DRIVE_HIGHWAYS ∪ PEDESTRIAN_SET ∪ CYCLE_SET ∪ {'path'}.
    '''

    highway = tags.get("highway")
    if highway is None:
        return False
    if tags.get("area") == "yes":
        return False
    if highway == "construction":
        return False
    if tags.get("access") == "private":
        return False
    return (
        (highway in DRIVE_HIGHWAYS)
        or (highway in PEDESTRIAN_SET)
        or (highway in CYCLE_SET)
        or (highway == "path")
    )
