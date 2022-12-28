import angiogenesis as an
import time
import numpy as np
import cProfile as profile

# In outer section of code
pr = profile.Profile()
pr.enable()



start_time = time.time()




hexagonal = 0
triangular = 0
number_of_rowscols = 39
number_of_nodes = number_of_rowscols*number_of_rowscols
incidence_matrix, graph, nodes_data, edges_data = an.generate_grid_graph(number_of_rowscols, number_of_rowscols, hexagonal=hexagonal, triangular=triangular)

source_value = number_of_rowscols - 1
source_list = an.localise_source(graph, source_value, corridor_model=0, two_capacitor_plates_model=0,
                                square_concentric_model=1, veins_square_concentric_model=0, triangular=0)

incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = \
    an.generate_physical_values(source_list, incidence_matrix)

arguments = {'pressure_list': pressure_list, 'length_list': length_list, 'conductivity_list': conductivity_list,
             'flow_list': flow_list, 'pressure_diff_list': pressure_diff_list, 'incidence_matrix': incidence_matrix,
             'incidence_T': incidence_T, 'incidence_inv': incidence_inv, 'graph': graph, 'source_list': source_list,
             'edges_data': edges_data, 'nodes_data': nodes_data, 'number_of_rowscols': number_of_rowscols, 'source_value': source_value}

an.set_graph_attributes(graph, pressure_list, conductivity_list, flow_list, pressure_diff_list)
an.update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time=True)

#print(edges_data)
#print(nodes_data)

# dK/dt = a*(Q/Q_hat)^(2*gamma) - b*K + c
parameters_set = {'a': 3.1, 'b': 4.5, 'gamma': 2/3, 'delta': 2.01, 'nu': 1.1, 'flow_hat': np.average(np.abs(flow_list)), 'c': 0.001, 'r': 2.2, 'dt': 0.01, 'N': 200}
an.run_simulation("lattice_53x53_N=160", **arguments, **parameters_set, is_scaled=True)

#print(edges_data)
#print(nodes_data)

print("time elapsed: {:.2f}s".format(time.time() - start_time))

pr.disable()
pr.dump_stats('profile3.pstat')

print("time elapsed after profiling: {:.2f}s".format(time.time() - start_time))

