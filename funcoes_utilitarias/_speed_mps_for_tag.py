from constantes.constantes import SPEED_LIMITS_MPS
from ._normalize_tag import _normalize_tag

def _speed_mps_for_tag(tag: str) -> float:
    '''
    Retorna a velocidade (m/s) associada à tag normalizada do tipo de via.
    
    Parâmetros
    ----------
    tag : str (tag original do CSV)

    Retorno
    ----------
    float : velocidade em m/s
    '''
    
    return SPEED_LIMITS_MPS.get(_normalize_tag(tag))
