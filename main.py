import os
import sys
import time
from cli.cli_stats import cli_stats
from cli.cli_nearest import cli_nearest
from cli.cli_dijkstra import cli_dijkstra
from cli.cli_astar import cli_astar
from cli.cli_astar_time import cli_astar_time
from cli.cli_vrp import cli_vrp
from cli.cli_node_to_xy_dist import cli_node_to_xy_dist

if __name__ == "__main__":
    prog = os.path.basename(__file__)
    usage = (
        "Uso:\n"
        f"  python {prog} stats <out_nodes_path> <out_edges_path>\n"
        f"  python {prog} nearest <out_nodes_path> <out_edges_path> <lat> <lon> <car|bike|foot>\n"
        f"  python {prog} dijkstra <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} astar  <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} astar_time  <out_nodes_path> <out_edges_path> <lat1> <lon1> <lat2> <lon2> <car|bike|foot>\n"
        f"  python {prog} vrp <out_nodes_path> <out_edges_path> <input_txt> <start_HH:MM> <car|bike|foot>\n"
        f"  python {prog} node_to_xy_dist <out_nodes_path> <out_edges_path> <x1> <y1> <node_id>\n"
    )
    start_time = time.time()
    if len(sys.argv) == 4 and sys.argv[1] == "stats":
        _, _, nodes_csv, edges_csv = sys.argv
        cli_stats(nodes_csv, edges_csv)
    elif len(sys.argv) == 7 and sys.argv[1] == "nearest":
        _, _, nodes_csv, edges_csv, lat, lon = sys.argv[:6]
        cli_nearest(nodes_csv, edges_csv, float(lat), float(lon), sys.argv[6])
    elif len(sys.argv) == 9 and sys.argv[1] == "dijkstra":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_dijkstra(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 9 and sys.argv[1] == "astar":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_astar(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 9 and sys.argv[1] == "astar_time":
        _, _, nodes_csv, edges_csv, lat1, lon1, lat2, lon2 = sys.argv[:8]
        cli_astar_time(nodes_csv, edges_csv, float(lat1), float(lon1), float(lat2), float(lon2), sys.argv[8])
    elif len(sys.argv) == 7 and sys.argv[1] == "vrp":
        _, _, nodes_csv, edges_csv, input_txt, start_hhmm = sys.argv[:6]
        cli_vrp(nodes_csv, edges_csv, input_txt, start_hhmm, sys.argv[6])
    elif len(sys.argv) == 7 and sys.argv[1] == "node_to_xy_dist":
        _, _, nodes_csv, edges_csv, x1, y1 = sys.argv[:6]
        node_id = int(sys.argv[6])
        cli_node_to_xy_dist(nodes_csv, edges_csv, float(x1), float(y1), node_id)
    else:
        print(usage, end="")
    execution_time = time.time() - start_time
    print("Tempo de execução total: %s segundos." % execution_time)
