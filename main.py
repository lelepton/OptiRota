from pathlib import Path
from osmgraph import build_road_network_graph, Router


def main():
    osm_file = Path("data\map.osm")

    if not osm_file.exists():
        print(f"Arquivo não encontrado: {osm_file}")
        print(
            "Por favor, baixe um arquivo .osm (ex: de https://www.openstreetmap.org/export)")
        return

    print(
        f"Construindo grafo a partir de '{osm_file}'... (Isso pode levar um tempo)")
    G = build_road_network_graph(osm_file)
    print(
        f"Grafo construído com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

    router = Router(G)

    lat_origem, lon_origem = -9.6684, -35.7032
    lat_destino, lon_destino = -9.6380, -35.7196

    print("\nCalculando a rota mais curta de carro...")

    # Encontra o caminho
    path_nodes, distance_meters = router.find_shortest_path(
        lat_origem, lon_origem,
        lat_destino, lon_destino,
        mode='car'  # pode ser 'car', 'bike' ou 'foot'
    )

    if path_nodes:
        print("Rota encontrada com sucesso!")
        print(f"Distância total: {distance_meters / 1000:.2f} km")
        print(f"A rota passa por {len(path_nodes)} nós.")
    else:
        print("Não foi possível encontrar uma rota.")


if __name__ == "__main__":
    main()
