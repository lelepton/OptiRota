import argparse
from .constantes import DEFAULT_OVERPASS_URL

def _build_arg_parser() -> argparse.ArgumentParser:
    '''
    Constrói e retorna o ArgumentParser da CLI.

    Parâmetros
    ----------
    (nenhum)

    Retorno
    -------
    argparse.ArgumentParser : parser configurado

    Opções expostas
    ---------------
    --center "lat,lon"  | --radius METROS | --nodes PATH | --edges PATH
    --overpass-url URL  | --save-osm PATH | --log-level LEVEL
    '''

    parser = argparse.ArgumentParser(
        prog="osm_circle_to_csv",
        description=(
            "Baixa OSM via Overpass para um círculo (lat,lon,R em metros) e gera dois CSVs:\n"
            " - nodes: osmid,y,x\n"
            " - edges: u,v,d,name,highway (d em metros)"
        ),
    )
    parser.add_argument(
        "--center",
        required=True,
        help="Coordenadas no formato 'lat,lon' (graus decimais). Ex.: -9.648123,-35.717456",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Raio em METROS para a busca circular (around). Ex.: 2000",
    )
    parser.add_argument("--nodes", dest="nodes_csv", required=True, help="Caminho do CSV de nós (saída)")
    parser.add_argument("--edges", dest="edges_csv", required=True, help="Caminho do CSV de arestas (saída)")
    parser.add_argument(
        "--overpass-url",
        default=DEFAULT_OVERPASS_URL,
        help=f"URL do endpoint Overpass (padrão: {DEFAULT_OVERPASS_URL})",
    )
    parser.add_argument(
        "--save-osm",
        dest="save_osm",
        default=None,
        help="Se fornecido, salva o .osm baixado neste caminho (útil para cache/inspeção).",
    )
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Nível de log (ex.: INFO, DEBUG)")
    return parser
