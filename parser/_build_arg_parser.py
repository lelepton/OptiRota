import argparse

# CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write_osm_csv_refactored",
        description=(
            "Converte um arquivo OSM (.osm) em dois CSVs:\n"
            " - nodes: osmid,y,x\n"
            " - edges: u,v,d,name,highway (d em metros)"
        ),
    )
    parser.add_argument("--in", dest="osm_in", required=True, help="Caminho do arquivo .osm de entrada")
    parser.add_argument("--nodes", dest="nodes_csv", required=True, help="Caminho do CSV de nós (saída)")
    parser.add_argument("--edges", dest="edges_csv", required=True, help="Caminho do CSV de arestas (saída)")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Nível de log (ex.: INFO, DEBUG)")
    return parser
