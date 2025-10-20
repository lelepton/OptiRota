import csv
from pathlib import Path
from typing import Dict, Tuple

def write_nodes_csv(nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    '''
    Gera o arquivo nodes.csv com cabeçalho (osmid, y, x), onde y=lat e x=lon.

    Parâmetros
    ----------
    nodes    : Dict[int, Tuple[float, float]]  -> {osmid: (lat, lon)}
    csv_path : Path                            -> caminho do CSV de saída

    Retorno
    -------
    None

    Complexidade
    ------------
    O(N log N) devido à ordenação dos IDs para saída determinística.
    '''

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["osmid", "y", "x"])
        for osmid in sorted(nodes.keys()):
            lat, lon = nodes[osmid]
            writer.writerow([osmid, f"{lat:.10f}", f"{lon:.10f}"])
            