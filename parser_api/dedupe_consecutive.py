from typing import Iterable, List

def dedupe_consecutive(ids_iter: Iterable[int]) -> List[int]:
    '''
    Remove duplicatas consecutivas de uma sequência de IDs, preservando a ordem.
    Útil para polilinhas com nós repetidos adjacentes.

    Parâmetros
    ----------
    ids_iter : Iterable[int]  -> sequência de IDs

    Retorno
    -------
    List[int] : sequência sem repetições consecutivas

    Complexidade
    ------------
    O(N), onde N é o comprimento da sequência.
    '''

    result: List[int] = []
    last: int | None = None
    for nid in ids_iter:
        if nid != last:
            result.append(nid)
            last = nid
    return result
