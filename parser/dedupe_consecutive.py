from typing import Iterable, List

def dedupe_consecutive(ids_iter: Iterable[int]) -> List[int]:
    '''
    Retorna uma lista sem IDs duplicados consecutivos. Isso evita arestas de
    comprimento zero quando o dado OSM repete o mesmo nó em sequência.

    Parâmetros
    ----------
    ids_iter : iterável de IDs de nós

    Retorno
    -------
    List[int] : nova lista sem duplicatas consecutivas.
    '''
    result: List[int] = []
    last: int | None = None
    for nid in ids_iter:
        if nid != last:
            result.append(nid)
            last = nid
    return result
