import Angiogenesis as an
import time
import networkx as nx
import numpy as np

start_time = time.time()

hexagonal = 0
triangular = 0
m = 13
number_of_nodes = m*m
incidence_matrix, graph, nodes_data, edges_data = an.generate_grid_graph(m, m, hexagonal=hexagonal, triangular=triangular)

source_value = m - 1
source_list = an.localise_source(graph, source_value, corridor_model=0, two_capacitor_plates_model=0,
                                square_concentric_model=1, veins_square_concentric_model=0, triangular=0)

incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = \
    an.generate_physical_values(source_list, incidence_matrix)

arguments = {'pressure_list': pressure_list, 'length_list': length_list, 'conductivity_list': conductivity_list,
             'flow_list': flow_list, 'pressure_diff_list': pressure_diff_list, 'incidence_matrix': incidence_matrix,
             'incidence_T': incidence_T, 'incidence_inv': incidence_inv, 'graph': graph, 'source_list': source_list,
             'incidence_T_inv': incidence_T_inv, 'x': x, 'x_dagger': x_dagger}

an.set_graph_attributes(graph, pressure_list, conductivity_list, flow_list, pressure_diff_list)
an.update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time=True)

# setting the layout for the graph visualisation
if hexagonal:
    pos = nx.get_node_attributes(graph, 'pos')  # hexagonal rigid layout
elif triangular:
    pos = nx.get_node_attributes(graph, 'pos')  # triangular rigid layout
else:
    pos = dict((n, n) for n in graph.nodes())   # square rigid layout


an.checking_Kirchhoffs_and_Murrays_law(graph, source_list)
an.draw_graph(graph, "initial_graph", pos, conductivity_list, m)
#print(edges_data)
#print(nodes_data)
print("Q_av:", np.average(np.abs(flow_list)))

# dK/dt = a*(Q/Q_hat)^(2*gamma) - b*K + c
parameters_set = {'a': 3.1, 'b': 4.5, 'gamma': 2/3, 'delta': 2.01, 'nu': 1.1, 'flow_hat': np.average(np.abs(flow_list)), 'c': 0.001, 'r': 2.2, 'dt': 0.01, 'N': 160}
graph, conductivity_list = an.run_simulation(source_value, m, pos, nodes_data, edges_data, **arguments, **parameters_set, is_scaled=True)

#print(edges_data)
#print(nodes_data)
an.draw_graph(graph, "final_graph", pos, conductivity_list, m)
an.checking_Kirchhoffs_and_Murrays_law(graph, source_list)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
