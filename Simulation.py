import Angiogenesis as an
import time
import networkx as nx
import numpy as np

start_time = time.time()

hexagonal = 0
triangular = 0

m = 20
number_of_nodes = m*m
#adjacency_matrix = an.generate_random_adjacent_matrix(number_of_nodes)
# random graph
#incidence_matrix, graph, nodes_data, edges_data = an.generate_graph(adjacency_matrix)

# generate lattice
incidence_matrix, graph, nodes_data, edges_data = an.generate_grid_graph(m, m, hexagonal=hexagonal, triangular=triangular)
#graph.remove

source_value = 10
incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = \
    an.generate_physical_values(graph, source_value, incidence_matrix, corridor_model=0, two_capacitor_plates_model=0, one_capacitor_plates_model=1, quater_model=0, triangular=triangular)

arguments = {'pressure_list': pressure_list, 'length_list': length_list, 'conductivity_list': conductivity_list,
             'flow_list': flow_list, 'pressure_diff_list': pressure_diff_list, 'incidence_matrix': incidence_matrix,
             'incidence_T': incidence_T, 'incidence_inv': incidence_inv, 'graph': graph, 'source_list': source_list,
             'incidence_T_inv': incidence_T_inv, 'x': x, 'x_dagger': x_dagger}

an.set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list)
an.update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time=True)

# setting the layout for the graph visualisation
if hexagonal:
    pos = nx.get_node_attributes(graph, 'pos')  # hexagonal rigid layout
elif triangular:
    pos = nx.get_node_attributes(graph, 'pos')  # triangular rigid layout
else:
    pos = dict((n, n) for n in graph.nodes())   # square rigid layout


an.checking_Kirchhoffs_law(graph, source_list)
an.draw_graph(graph, "graph", pos, conductivity_list, m)
#print(edges_data)
# dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c
parameters_set = {'a': 2.9, 'b': 1.3, 'gamma': 2/3, 'delta': 1.1, 'nu': 1.1, 'flow_hat': 1.1, 'c': 0.000001, 'r': 2, 'dt': 0.1, 'N': 200}

graph, conductivity_list = an.run_simulation(m, pos, nodes_data, edges_data, **arguments, **parameters_set, is_scaled=True, with_pruning=False)
#print(edges_data)
an.draw_graph(graph, "final_graph", pos, conductivity_list, m)
an.checking_Kirchhoffs_law(graph, source_list)


print("time elapsed: {:.2f}s".format(time.time() - start_time))
