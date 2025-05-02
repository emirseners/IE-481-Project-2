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

        # To save random graphs as txt files
        txt_path = Path("inputs/random") / f"graph_{graph_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for each_row in adjacency_matrix:
                f.write(''.join(map(str, each_row)) + '\n')

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
    colors = [1 if vertex_degrees[i] == 0 else 0 for i in range(graph_order)]

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
    model.setParam('OutputFlag', False)

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
    model.setParam('OutputFlag', False)

    x_i = model.addVars([i for i in range(len(maximal_independent_sets))], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

    model.setObjective(quicksum(x_i[i] for i in range(len(maximal_independent_sets))), GRB.MINIMIZE)

    for vertex in range(graph_order):
        model.addConstr(sum(x_i[set_index] for set_index, independent_set in enumerate(maximal_independent_sets) if vertex in independent_set) >= 1)

    model.optimize()

    chosen_sets = [maximal_independent_sets[set_index] for set_index in x_i if x_i[set_index].X >= 1]

    lp_colors = [0] * graph_order
    for colour_id, ind_set in enumerate(chosen_sets, 1):
        for v in ind_set:
            lp_colors[v] = colour_id

    return lp_colors, len(chosen_sets), model.Runtime

def check_if_proper_and_maximal_coloring(adjacency_matrix, colors):    #Returns 1 if the coloring is proper, 0 otherwise
    if (all(colors[i] != colors[j] for i in range(len(adjacency_matrix)) for j in range(i+1, len(adjacency_matrix)) if adjacency_matrix[i][j]) and all(c in range(1, len(adjacency_matrix) + 1) for c in colors)):
        return 1
    else:
        return 0

result = {}
proper_coloring_count = {'greedy': 0, 'largest_first': 0, 'dsatur': 0, 'ip': 0, 'lp_relaxation': 0}

for graph_name, adjacency_matrix in graph_inputs.items():
    greedy_colors, greedy_result, greedy_time = greedy_algorithm(adjacency_matrix)
    largest_first_colors, largest_first_result, largest_first_time = largest_first_algorithm(adjacency_matrix)
    dsatur_colors, dsatur_result, dsatur_time = dsatur_algorithm(adjacency_matrix)
    ip_colors, ip_result, ip_time = integer_programming_model(adjacency_matrix)
    lp_relaxation_colors, lp_relaxation_result, lp_relaxation_time = lp_relaxation_model(adjacency_matrix)

    result[graph_name] = [greedy_colors, greedy_result, greedy_result/ip_result, greedy_time, 
                          largest_first_colors, largest_first_result, largest_first_result/ip_result, largest_first_time,
                          dsatur_colors, dsatur_result, dsatur_result/ip_result, dsatur_time,
                          ip_colors, ip_result, ip_time,
                          lp_relaxation_colors, lp_relaxation_result, lp_relaxation_time]
    
    proper_coloring_count['greedy'] += check_if_proper_and_maximal_coloring(adjacency_matrix, greedy_colors)
    proper_coloring_count['largest_first'] += check_if_proper_and_maximal_coloring(adjacency_matrix, largest_first_colors)
    proper_coloring_count['dsatur'] += check_if_proper_and_maximal_coloring(adjacency_matrix, dsatur_colors)
    proper_coloring_count['ip'] += check_if_proper_and_maximal_coloring(adjacency_matrix, ip_colors)
    proper_coloring_count['lp_relaxation'] += check_if_proper_and_maximal_coloring(adjacency_matrix, lp_relaxation_colors)

question_1a_by_type = {}
question_1a_by_order = {}
question_1a_by_density = {}

for key, item in result.items():
    txt_name_parts = key.split('_')
    agg_key_by_type = f"{txt_name_parts[0]}"
    agg_key_by_order = f"{txt_name_parts[1]}"
    agg_key_by_density = f"{txt_name_parts[2]}"

    if txt_name_parts[0] == 'perfect':
        continue

    greedy_opt_ratio = item[2]
    largest_first_opt_ratio = item[6]
    dsatur_opt_ratio = item[10]

    if agg_key_by_type not in question_1a_by_type:
        question_1a_by_type[agg_key_by_type] = {'greedy': 0, 'largest_first': 0, 'dsatur': 0}

    if agg_key_by_order not in question_1a_by_order:
        question_1a_by_order[agg_key_by_order] = {'greedy': 0, 'largest_first': 0, 'dsatur': 0}

    if agg_key_by_density not in question_1a_by_density:
        question_1a_by_density[agg_key_by_density] = {'greedy': 0, 'largest_first': 0, 'dsatur': 0}

    question_1a_by_type[agg_key_by_type]['greedy'] += greedy_opt_ratio
    question_1a_by_type[agg_key_by_type]['largest_first'] += largest_first_opt_ratio
    question_1a_by_type[agg_key_by_type]['dsatur'] += dsatur_opt_ratio

    question_1a_by_order[agg_key_by_order]['greedy'] += greedy_opt_ratio
    question_1a_by_order[agg_key_by_order]['largest_first'] += largest_first_opt_ratio
    question_1a_by_order[agg_key_by_order]['dsatur'] += dsatur_opt_ratio

    question_1a_by_density[agg_key_by_density]['greedy'] += greedy_opt_ratio
    question_1a_by_density[agg_key_by_density]['largest_first'] += largest_first_opt_ratio
    question_1a_by_density[agg_key_by_density]['dsatur'] += dsatur_opt_ratio


def adjust_scale(metrics_dict, divisor):
    for inner in metrics_dict.values():
        for k in inner:
            inner[k] /= divisor

adjust_scale(question_1a_by_type, 225)
adjust_scale(question_1a_by_order, 30)
adjust_scale(question_1a_by_density, 150)

question_1b = {}

for key, item in result.items():
    txt_name_parts = key.split('_')
    agg_key = f"{txt_name_parts[0]}"

    if txt_name_parts[0] == 'perfect':
        continue

    adjacency_matrix = graph_inputs[key]
    degree_of_each_vertex = [sum(row) for row in adjacency_matrix]
    max_degree = max(degree_of_each_vertex)

    result_greedy = item[1]

    hits, average_gap, amount = question_1b.get(agg_key, (0, 0, 0))

    if result_greedy == max_degree + 1:
        hits += 1

    current_average_gap = (average_gap * amount + (100 * (max_degree + 1 - result_greedy) / (max_degree + 1))) / (amount + 1)

    amount += 1
    question_1b[agg_key] = (hits, current_average_gap, amount)

question_1c = {}

for key, item in result.items():
    txt_name_parts = key.split('_')
    agg_key = f"{txt_name_parts[0]}"

    if txt_name_parts[0] == 'perfect':
        continue

    adjacency_matrix = graph_inputs[key]
    max_min_value = max([min(index+1, sum(row)+1) for index, row in enumerate(adjacency_matrix)])

    result_largest_first = item[5]

    hits, average_gap, amount = question_1c.get(agg_key, (0, 0, 0))

    if result_largest_first == max_min_value:
        hits += 1

    current_average_gap = (average_gap * amount + (100 * (max_min_value - result_largest_first) / (max_min_value))) / (amount + 1)

    amount += 1
    question_1c[agg_key] = (hits, current_average_gap, amount)

question_2a = {}

for key, item in result.items():
    txt_name_parts = key.split('_')
    agg_key = f"{txt_name_parts[0]}"

    if agg_key not in question_2a:
        question_2a[agg_key] = {'number_of_optimal': 0, 'number_of_not_optimal': 0, 'percentage': 0}

    dsatur_opt_ratio = item[10]

    if dsatur_opt_ratio == 1:
        question_2a[agg_key]['number_of_optimal'] += 1
    else:
        question_2a[agg_key]['percentage'] = (question_2a[agg_key]['percentage'] * question_2a[agg_key]['number_of_not_optimal'] + (dsatur_opt_ratio * 100)) / (question_2a[agg_key]['number_of_not_optimal'] + 1)
        question_2a[agg_key]['number_of_not_optimal'] += 1

question_2b = {}

for key, item in result.items():
    txt_name_parts = key.split('_')
    agg_key = f"{txt_name_parts[0]}"

    best_greedy_result = item[9]
    lp_result = item[13]
    lp_relaxation_result = item[16]

    best_greedy_running_time = item[11]
    lp_running_time = item[14]
    lp_relaxation_running_time = item[17]

    if agg_key not in question_2b:
        question_2b[agg_key] = {'best_greedy_result': 0, 'lp_result': 0, 'lp_relaxation_result': 0, 
                                'best_greedy_running_time': 0, 'lp_running_time': 0, 'lp_relaxation_running_time': 0, 'count': 0}

    question_2b[agg_key]['best_greedy_result'] = (question_2b[agg_key]['best_greedy_result'] * question_2b[agg_key]['count'] + best_greedy_result) / (question_2b[agg_key]['count'] + 1)
    question_2b[agg_key]['lp_result'] = (question_2b[agg_key]['lp_result'] * question_2b[agg_key]['count'] + lp_result) / (question_2b[agg_key]['count'] + 1)
    question_2b[agg_key]['lp_relaxation_result'] = (question_2b[agg_key]['lp_relaxation_result'] * question_2b[agg_key]['count'] + lp_relaxation_result) / (question_2b[agg_key]['count'] + 1)

    question_2b[agg_key]['best_greedy_running_time'] = (question_2b[agg_key]['best_greedy_running_time'] * question_2b[agg_key]['count'] + best_greedy_running_time) / (question_2b[agg_key]['count'] + 1)
    question_2b[agg_key]['lp_running_time'] = (question_2b[agg_key]['lp_running_time'] * question_2b[agg_key]['count'] + lp_running_time) / (question_2b[agg_key]['count'] + 1)
    question_2b[agg_key]['lp_relaxation_running_time'] = (question_2b[agg_key]['lp_relaxation_running_time'] * question_2b[agg_key]['count'] + lp_relaxation_running_time) / (question_2b[agg_key]['count'] + 1)

    question_2b[agg_key]['count'] += 1

selected_indices_2c = [1, 2, 3, 5, 6, 7, 9, 10, 11]
question_2c = {key: [value[i] for i in selected_indices_2c] for key, value in result.items() if key[:7] == 'perfect'}

selected_indices_2d = [15, 16, 17]
question_2d = {key: [value[i] for i in selected_indices_2d] for key, value in result.items()}

selected_indices_2e = [12, 13, 14, 15, 16, 17]
question_2e = {key: [value[i] for i in selected_indices_2e] + [value[16] / value[13]] for key, value in result.items()}

result_cols = [
    'greedy_colors', 'greedy_result', 'greedy_opt_ratio', 'greedy_time',
    'largest_first_colors', 'largest_first_result', 'largest_first_opt_ratio', 'largest_first_time',
    'dsatur_colors', 'dsatur_result', 'dsatur_opt_ratio', 'dsatur_time',
    'ip_colors', 'ip_result', 'ip_time',
    'lp_relaxation_colors', 'lp_relaxation_result', 'lp_relaxation_time'
]

question_2c_cols = [
    'greedy_result', 'greedy_opt_ratio', 'greedy_time',
    'largest_first_result', 'largest_first_opt_ratio', 'largest_first_time',
    'dsatur_result', 'dsatur_opt_ratio', 'dsatur_time',
]

question_2d_cols = ['lp_relaxation_colors', 'lp_relaxation_result', 'lp_relaxation_time']

question_2e_cols = [
    'ip_colors', 'ip_result', 'ip_time',
    'lp_relaxation_colors', 'lp_relaxation_result', 'lp_relaxation_time',
    'lp_over_ip_ratio'
]

result_df = pd.DataFrame.from_dict(result, orient='index', columns=result_cols)
result_df.index.name = 'graph'

q1a_type_df    = pd.DataFrame.from_dict(question_1a_by_type, orient='index')
q1a_order_df   = pd.DataFrame.from_dict(question_1a_by_order, orient='index')
q1a_density_df = pd.DataFrame.from_dict(question_1a_by_density, orient='index')
q1a_type_df.index.name    = 'graph_type'
q1a_order_df.index.name   = 'graph_order'
q1a_density_df.index.name = 'graph_density'

q1b_df = pd.DataFrame.from_dict(question_1b, orient='index', columns=['hits', 'average_gap', 'amount']).rename_axis('graph_type')

q1c_df = pd.DataFrame.from_dict(question_1c, orient='index', columns=['hits', 'average_gap', 'amount']).rename_axis('graph_type')

q2a_df = pd.DataFrame.from_dict(question_2a, orient='index')
q2a_df.index.name = 'graph_type'

q2b_df = pd.DataFrame.from_dict(question_2b, orient='index')
q2b_df.index.name = 'graph_type'

q2c_df = pd.DataFrame.from_dict(question_2c, orient='index', columns=question_2c_cols)
q2c_df.index.name = 'graph'

q2d_df = pd.DataFrame.from_dict(question_2d, orient='index', columns=question_2d_cols)
q2d_df.index.name = 'graph'

q2e_df = pd.DataFrame.from_dict(question_2e, orient='index', columns=question_2e_cols)
q2e_df.index.name = 'graph'

proper_df = (pd.Series(proper_coloring_count, name='proper_colorings').to_frame())
proper_df.index.name = 'algorithm'

output_path = Path('graph_coloring_analysis.xlsx')
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    result_df.to_excel(writer,          sheet_name='result')

    q1a_type_df.to_excel(writer, sheet_name='q1a_type')
    q1a_order_df.to_excel(writer, sheet_name='q1a_order')
    q1a_density_df.to_excel(writer, sheet_name='q1a_density')
    q1b_df.to_excel(writer, sheet_name='q1b')
    q1c_df.to_excel(writer, sheet_name='q1c')

    q2a_df.to_excel(writer, sheet_name='q2a')
    q2b_df.to_excel(writer, sheet_name='q2b')
    q2c_df.to_excel(writer, sheet_name='q2c')
    q2d_df.to_excel(writer, sheet_name='q2d')
    q2e_df.to_excel(writer, sheet_name='q2e')

    proper_df.to_excel(writer, sheet_name='proper_coloring_count')

print('Done!!!')