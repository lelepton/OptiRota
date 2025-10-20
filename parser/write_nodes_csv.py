import csv
from pathlib import Path
from typing import Dict, Tuple

def write_nodes_csv(nodes: Dict[int, Tuple[float, float]], csv_path: Path) -> None:
    '''
    Escreve o CSV de nós com colunas (osmid, y, x), onde y = latitude e
    x = longitude. A saída é ordenada pelo ID do nó para garantir determinismo.

    Parâmetros
    ----------
    nodes    : mapeamento osmid -> (lat, lon)
    csv_path : caminho do arquivo CSV de saída
    '''

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["osmid", "y", "x"])
        for osmid in sorted(nodes.keys()):
            lat, lon = nodes[osmid]
            writer.writerow([osmid, f"{lat:.10f}", f"{lon:.10f}"])
            