import Angiogenesis as an
import time
import networkx as nx

start_time = time.time()

n = 3
number_of_nodes = n*n
adjacency_matrix = an.generate_random_adjacent_matrix(number_of_nodes)
#incidence_matrix, graph, nodes_data, edges_data = generate_graph(adjacency_matrix)     # random graph
incidence_matrix, graph, nodes_data, edges_data = an.generate_grid_graph(n, n)          # regular grid
source_value = 5
x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = \
    an.generate_physical_values(number_of_nodes, source_value, incidence_matrix)

arguments = {'pressure_list': pressure_list, 'length_list': length_list, 'conductivity_list': conductivity_list,
             'flow_list': flow_list, 'pressure_diff_list': pressure_diff_list, 'incidence_matrix': incidence_matrix,
             'incidence_T': incidence_T, 'incidence_inv': incidence_inv, 'graph': graph, 'source_list': source_list}

an.set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list)

an.update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time=True)
pos = nx.spring_layout(graph)
#pos_x = nx.multipartite_layout(graph, subset_key="layer")
print(graph)
an.draw_graph(graph, "graph", pos, conductivity_list)
# dK/dt = a*(exp(r*t/2)^(delta) * q / q_hat)^(2*gamma) - b * K + c
parameters_set = {'a': 1.1, 'b': 1.1, 'gamma': 1.1, 'delta': 1.1, 'flow_hat': 1.1, 'c': 10.1, 'r': 1.1, 'dt': 0.0001, 'N': 20000}
an.run_simulation_A(**arguments, **parameters_set)
#an.run_simulation_G(graph, **parameters_set)
an.draw_graph(graph, "final_graph", pos, conductivity_list)
print(graph)
an.update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
