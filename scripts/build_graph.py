from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from osmgraph.graph_builder import build_graph_in_memory

if __name__ == "__main__":
    osm_file = Path(__file__).resolve().parent.parent / "data" / "map.osm"
    G = build_graph_in_memory(osm_file)

    print(f"Nós: {G.number_of_nodes()}, Arestas: {G.number_of_edges()}")

    pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=5, edge_color='gray', arrowsize=5)
    plt.show()
