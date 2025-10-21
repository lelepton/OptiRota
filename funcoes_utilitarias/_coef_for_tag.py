from constantes.constantes import COEF_BY_TAG
from ._normalize_tag import _normalize_tag

def _coef_for_tag(tag: str) -> float:
    '''
    Retorna o coeficiente usado no custo artificial da A* com tempo,
    associado à tag normalizada do tipo de via.

    Parâmetros
    ----------
    tag : str (tag original do CSV)

    Retorno
    ----------
    float : coeficiente (adimensional)
    '''
    
    return COEF_BY_TAG.get(_normalize_tag(tag))
