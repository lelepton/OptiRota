from dataclasses import dataclass

@dataclass(frozen=True)
class Edge:
    '''
    Aresta dirigida u->v com:
    - v (destino), w (peso 'd' em metros), tag (highway)
    - flags de acesso: allow_car, allow_bike, allow_foot
    - name: nome da via (pode ser "")

    Observações
    -----------
    Dataclass imutável (frozen=True) para facilitar depuração e segurança.
    '''
    v: int
    w: float
    tag: str
    allow_car: bool
    allow_bike: bool
    allow_foot: bool
    name: str