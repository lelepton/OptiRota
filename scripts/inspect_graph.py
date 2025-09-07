from pathlib import Path
from osmgraph.graph_builder import build_graph_in_memory

if __name__ == "__main__":
    osm_file = Path("../data/exemplo.osm")
    G = build_graph_in_memory(osm_file)

    print(f"Nós: {G.number_of_nodes()}, Arestas: {G.number_of_edges()}\n")

    print("Exemplo de nós:")
    for n, data in list(G.nodes(data=True))[:5]:
        print(f"Nó {n}: {data}")

    print("\nExemplo de arestas:")
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"Aresta {u} -> {v}: {data}")
