import Angiogenesis as an
import time
import networkx as nx

start_time = time.time()

hexagonal = False
triangular = False
n = 7
number_of_nodes = n*n
adjacency_matrix = an.generate_random_adjacent_matrix(number_of_nodes)
# random graph
#incidence_matrix, graph, nodes_data, edges_data = generate_graph(adjacency_matrix)

# generate lattice
incidence_matrix, graph, nodes_data, edges_data = an.generate_grid_graph(n, n, hexagonal=hexagonal, triangular=triangular)
source_value = 10
x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = \
    an.generate_physical_values(source_value, incidence_matrix)

arguments = {'pressure_list': pressure_list, 'length_list': length_list, 'conductivity_list': conductivity_list,
             'flow_list': flow_list, 'pressure_diff_list': pressure_diff_list, 'incidence_matrix': incidence_matrix,
             'incidence_T': incidence_T, 'incidence_inv': incidence_inv, 'graph': graph, 'source_list': source_list}

an.set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list)

#an.update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time=True)

if hexagonal:
    pos = nx.get_node_attributes(graph, 'pos')  # hexagonal rigid layout
elif triangular:
    pos = nx.get_node_attributes(graph, 'pos')  # triangular rigid layout
else:
    pos = nx.spring_layout(graph)                # standard layout


an.checking_Kirchhoffs_law(graph, source_list)

an.draw_graph(graph, "graph", pos, conductivity_list, n)
# dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c
parameters_set = {'a': 1.1, 'b': 1.1, 'gamma': 2/3, 'delta': 0, 'flow_hat': 1.1, 'c': 0.1, 'r': 0, 'dt': 0.1, 'N': 100}
an.run_simulation_A(nodes_data, edges_data, **arguments, **parameters_set)
an.draw_graph(graph, "final_graph", pos, conductivity_list, n)

an.checking_Kirchhoffs_law(graph, source_list)


print("time elapsed: {:.2f}s".format(time.time() - start_time))
