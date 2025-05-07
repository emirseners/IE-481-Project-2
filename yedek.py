from gurobipy import Model, GRB, quicksum, tuplelist
from pathlib import Path
import networkx as nx
import pandas as pd
import random
import time

graph_inputs = {}
folders = Path("inputs/p4free"), Path("inputs/perfect")

for folder in folders:
    for txt_file in folder.glob("*.txt"):
        with txt_file.open(encoding="utf-8") as f:
            adjacency_matrix = [list(map(int, line.strip())) for line in f]
        adjusted_file_name = txt_file.name.removeprefix("graph_").replace("operations_", "").removesuffix(".txt")
        graph_inputs[adjusted_file_name] = adjacency_matrix

unique_order_density = {tuple(Path(fname).stem.split('_')[1:3]) for fname in graph_inputs}    #15 graph orders, 3 densities each, 5 replications each (which we obtained by printing)

def generate_erdos_renyi_graph(order, density, seed=None):
    if seed is not None:
        random.seed(seed)

    adjacency_matrix = [[0] * order for _ in range(order)]
    for i in range(order-1):
        for j in range(i+1, order):
            if random.random() < density:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix

for order, density in unique_order_density:
    for iteration_no in range(5):
        graph_name = f"random_{order}_{density}_0000{iteration_no+1}"
        order_input = int(order)
        density_input = float(density)/100000
        adjacency_matrix = generate_erdos_renyi_graph(order_input, density_input)
        graph_inputs[graph_name] = adjacency_matrix

for graph_name, adjacency_matrix in graph_inputs.items():
    txt_name_parts = graph_name.split('_')
    agg_key_by_type = f"{txt_name_parts[0]}"
    agg_key2_by_type = f"{txt_name_parts[1]}"

    if agg_key_by_type == "perfect":
        if agg_key2_by_type == "00050":

            print(f"Graph: {graph_name}")

            graph_order = len(adjacency_matrix)

            input_graph = nx.Graph()
            input_graph.add_nodes_from(range(graph_order))
            input_graph.add_edges_from((i, j) for i in range(graph_order-1) for j in range(i + 1, graph_order) if adjacency_matrix[i][j] == 1)

            complement_graph = nx.complement(input_graph)
            maximal_independent_sets = list(nx.find_cliques(complement_graph))
            print(f"maximal_independent_sets: {maximal_independent_sets}")