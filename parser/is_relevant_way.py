from typing import Dict
from .constantes import DRIVE_HIGHWAYS, PEDESTRIAN_SET, CYCLE_SET

def is_relevant_way(tags: Dict[str, str]) -> bool:
    '''
    Define se uma way deve ser incluída no grafo exportado.
    Exclui áreas, trechos em construção e acesso privado.
    Inclui vias dirigíveis, vias de pedestres, ciclovias e 'path'.

    Parâmetros
    ----------
    tags : Dict[str, str] com as tags da way

    Retorno
    -------
    bool : True se a way for relevante; False caso contrário.
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
