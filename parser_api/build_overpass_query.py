def build_overpass_query(lat: float, lon: float, radius_m: float) -> str:
    '''
    Monta a consulta Overpass QL (saída XML) que seleciona ways com 'highway'
    dentro de um círculo definido por (lat, lon) e raio (em METROS), e usa a
    recursão '>' para incluir os nós.
    
    Parâmetros
    ----------
    lat, lon  : float  -> centro do círculo (graus decimais)
    radius_m  : float  -> raio em METROS
    
    Retorno
    -------
    str : consulta Overpass QL formatada

    Complexidade
    ------------
    O(1) — apenas construção de string.
    
    Observações
    -----------
    - Usa '[out:xml]' para aproveitar o parser XML existente.
    - Timeout ampliado (180s) para áreas densas.
    '''

    q = f"""
[out:xml][timeout:180];
(
  way[highway](around:{int(radius_m)},{lat:.7f},{lon:.7f});
);
(._;>;);
out body;
"""
    return "\n".join(line.strip() for line in q.strip().splitlines())
