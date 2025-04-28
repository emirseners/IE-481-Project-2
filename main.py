from gurobipy import Model, GRB, quicksum, tuplelist
from pathlib import Path
import networkx as nx
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

        """
        # To save random graphs as txt files
        txt_path = Path("inputs/random") / f"graph_{graph_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for each_row in adjacency_matrix:
                f.write(''.join(map(str, each_row)) + '\n')
        """

def greedy_algorithm(adjacency_matrix):
    start_time = time.time()

    graph_order = len(adjacency_matrix)
    random_order = list(range(graph_order))
    random.shuffle(random_order)
    colors = [0] * graph_order

    for vertex_index, vertex in enumerate(random_order):

        colors_of_neighbors = {colors[neighbor_vertex] for neighbor_vertex in random_order[:vertex_index] if adjacency_matrix[vertex][neighbor_vertex] == 1}

        c = 1
        while c in colors_of_neighbors:
            c += 1
        colors[vertex] = c

    return colors, max(colors), time.time() - start_time

def largest_first_algorithm(adjacency_matrix):
    start_time = time.time()

    graph_order = len(adjacency_matrix)
    vertex_degrees = [sum(each_row) for each_row in adjacency_matrix]
    non_increasing_degree_order = sorted(range(graph_order), key=lambda x: vertex_degrees[x], reverse=True)
    colors = [0] * graph_order

    for vertex_index, vertex in enumerate(non_increasing_degree_order):

        colors_of_neighbors = {colors[neighbor_vertex] for neighbor_vertex in non_increasing_degree_order[:vertex_index] if adjacency_matrix[vertex][neighbor_vertex] == 1}

        c = 1
        while c in colors_of_neighbors:
            c += 1
        colors[vertex] = c

    return colors, max(colors), time.time() - start_time

def dsatur_algorithm(adjacency_matrix):
    start_time = time.time()

    graph_order = len(adjacency_matrix)
    vertex_degrees = [sum(each_row) for each_row in adjacency_matrix]
    colors = [0] * graph_order

    while 0 in colors:
        dsatur_list = [0] * graph_order
        for vertex_index in range(graph_order):
            if colors[vertex_index] == 0:
                dsatur_list[vertex_index] = len({colors[neighbor_index] for neighbor_index in range(graph_order) if (adjacency_matrix[vertex_index][neighbor_index] == 1 and colors[neighbor_index] != 0)})

        max_dsatur = max(dsatur_list)
        vertex_indices_with_max_dsatur = [index_with_max for index_with_max, dsatur in enumerate(dsatur_list) if dsatur == max_dsatur]
        vertex_indices_with_max_dsatur.sort(key=lambda x: vertex_degrees[x], reverse=True)
        vertex = vertex_indices_with_max_dsatur[0]

        colors_of_neighbors = {colors[neighbor_vertex] for neighbor_vertex in range(graph_order) if adjacency_matrix[vertex][neighbor_vertex] == 1 and colors[neighbor_vertex] != 0}

        c = 1
        while c in colors_of_neighbors:
            c += 1
        colors[vertex] = c

    return colors, max(colors), time.time() - start_time

def integer_programming_model(adjacency_matrix):
    graph_order = len(adjacency_matrix)

    input_graph = nx.Graph()
    input_graph.add_nodes_from(range(graph_order))
    input_graph.add_edges_from((i, j) for i in range(graph_order-1) for j in range(i + 1, graph_order) if adjacency_matrix[i][j] == 1)

    complement_graph = nx.complement(input_graph)
    maximal_independent_sets = list(nx.find_cliques(complement_graph))

    model = Model("IP_Vertex_Coloring")
    model.setParam('OutputFlag', True)

    x_i = model.addVars([i for i in range(len(maximal_independent_sets))], vtype=GRB.BINARY, name="x")

    model.setObjective(quicksum(x_i[i] for i in range(len(maximal_independent_sets))), GRB.MINIMIZE)

    for vertex in range(graph_order):
        model.addConstr(sum(x_i[set_index] for set_index, independent_set in enumerate(maximal_independent_sets) if vertex in independent_set) >= 1)

    model.optimize()

    chosen_sets = [maximal_independent_sets[set_index] for set_index in x_i if x_i[set_index].X > 0.5]

    colors = [0] * graph_order
    for color_index, independent_set in enumerate(chosen_sets):
        for vertex in independent_set:
            colors[vertex] = color_index + 1

    return colors, len(chosen_sets), model.Runtime

def lp_relaxation_model(adjacency_matrix):
    graph_order = len(adjacency_matrix)

    input_graph = nx.Graph()
    input_graph.add_nodes_from(range(graph_order))
    input_graph.add_edges_from((i, j) for i in range(graph_order-1) for j in range(i + 1, graph_order) if adjacency_matrix[i][j] == 1)

    complement_graph = nx.complement(input_graph)
    maximal_independent_sets = list(nx.find_cliques(complement_graph))

    model = Model("LP_Relaxation_Vertex_Coloring")
    model.setParam('OutputFlag', True)

    x_i = model.addVars([i for i in range(len(maximal_independent_sets))], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

    model.setObjective(quicksum(x_i[i] for i in range(len(maximal_independent_sets))), GRB.MINIMIZE)

    for vertex in range(graph_order):
        model.addConstr(sum(x_i[set_index] for set_index, independent_set in enumerate(maximal_independent_sets) if vertex in independent_set) >= 1)

    model.optimize()

    chosen_sets = [maximal_independent_sets[set_index] for set_index in x_i if x_i[set_index].X > 0.5]

    lp_solution = [(set_index, maximal_independent_sets[set_index], x_i[set_index].X) for set_index in range(len(maximal_independent_sets)) if x_i[set_index].X > 1e-6]

    return lp_solution, len(chosen_sets), model.Runtime

def check_if_propoer_coloring(adjacency_matrix, colors):
    return all(colors[i] != colors[j] for i in range(len(adjacency_matrix)) for j in range(i+1, len(adjacency_matrix)) if adjacency_matrix[i][j])

result = {}

for graph_name, adjacency_matrix in graph_inputs.items():
    greedy_colors, greedy_chromatic_number, greedy_time = greedy_algorithm(adjacency_matrix)
    largest_first_colors, largest_first_chromatic_number, largest_first_time = largest_first_algorithm(adjacency_matrix)
    dsatur_colors, dsatur_chromatic_number, dsatur_time = dsatur_algorithm(adjacency_matrix)
    ip_colors, ip_chromatic_number, ip_time = integer_programming_model(adjacency_matrix)
    lp_relaxation_colors, lp_relaxation_chromatic_number, lp_relaxation_time = lp_relaxation_model(adjacency_matrix)

    result[graph_name] = [greedy_colors, greedy_chromatic_number, greedy_time, 
                          largest_first_colors, largest_first_chromatic_number, largest_first_time,
                          dsatur_colors, dsatur_chromatic_number, dsatur_time,
                          ip_colors, ip_chromatic_number, ip_time,
                          lp_relaxation_colors, lp_relaxation_chromatic_number, lp_relaxation_time]

