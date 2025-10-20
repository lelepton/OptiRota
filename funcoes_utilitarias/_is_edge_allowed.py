from classes_de_elementos.edge import Edge

def _is_edge_allowed(travel_mode: str, edge: "Edge") -> bool:
    '''
    Informa se uma aresta é permitida para o modo de viagem informado.

    Parâmetros
    ----------
    travel_mode : str ('car' | 'bike' | 'foot')
    edge        : Edge

    Retorno
    ----------
    bool : True se a aresta pode ser usada no modo; False caso contrário
    '''

    mode = travel_mode.lower()
    if mode == "car":
        return edge.allow_car
    if mode == "bike":
        return edge.allow_bike or edge.allow_car
    # foot
    return edge.allow_foot or edge.allow_bike or edge.allow_car