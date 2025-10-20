import csv
from pathlib import Path
from typing import Dict, Tuple
from iter_ways import iter_ways
from is_relevant_way import is_relevant_way
from dedupe_consecutive import dedupe_consecutive
from compute_haversine_meters import compute_haversine_meters

def write_edges_csv(osm_path: Path, nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    '''
    Gera o arquivo edges.csv com cabeçalho (u, v, d, name, highway). Para cada
    par consecutivo de nós (u, v) em uma way relevante, escreve arestas em
    conformidade com 'oneway'. O campo d é a distância Haversine em METROS.
    
    Parâmetros
    ----------
    osm_path : Path                             -> caminho do arquivo .osm (XML)
    nodes    : Dict[int, Tuple[float, float]]   -> {osmid: (lat, lon)}
    csv_path : Path                             -> caminho do CSV de arestas
    
    Retorno
    -------
    None
    
    Complexidade
    ------------
    O(W * K), onde W é o número de ways relevantes e K o número médio de
    segmentos (pares consecutivos de nós) por way.
    
    Observações
    -----------
    - Aplica dedupe_consecutive e ignora nós ausentes na tabela de nós.
    - oneway in {'yes','true','1'} -> (u→v)
    - oneway == '-1'               -> (v→u)
    - caso contrário               -> (u→v) e (v→u)
    '''

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "d", "name", "highway"])

        for node_ids, tags in iter_ways(osm_path):
            if not is_relevant_way(tags):
                continue

            name = tags.get("name", "") or ""
            highway = tags.get("highway", "") or ""
            oneway = (tags.get("oneway", "no") or "").lower()

            clean_ids = [nid for nid in dedupe_consecutive(node_ids) if nid in nodes]
            if len(clean_ids) < 2:
                continue

            def emit(a: int, b: int, dist_m: float) -> None:
                writer.writerow([a, b, f"{dist_m:.3f}", name, highway])

            for u, v in zip(clean_ids[:-1], clean_ids[1:]):
                lat_u, lon_u = nodes[u]
                lat_v, lon_v = nodes[v]
                d_m = compute_haversine_meters(lat_u, lon_u, lat_v, lon_v)

                if oneway in {"yes", "true", "1"}:
                    emit(u, v, d_m)
                elif oneway == "-1":
                    emit(v, u, d_m)
                else:
                    emit(u, v, d_m)
                    emit(v, u, d_m)
