import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy

"""
General idea:
adjacency matrix --> incidence matrix, nodes, edges --> 
---> assign physical attributes --> run simulation --> create visualisation
"""


# undirected graph
def generate_random_adjacent_matrix(dimension):   # dimension == #nodes
    condition = True
    while condition:
        matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    matrix[i][j] = random.choice([0, 1])
        L = np.tril(matrix)
        U = np.transpose(L)                    # np.sum(U) == #edges
        symmetric_matrix = L + U
        if np.sum(symmetric_matrix) != 0:
            if np.trace(symmetric_matrix) == 0:
                if np.linalg.det(symmetric_matrix) != 0:
                    return symmetric_matrix
                    condition = False        # finish the process and create required matrix
                else:
                    pass
                    #print("Error: some of the nodes are not connected.")
            else:
                pass
                #print("Error: generated matrix isn't an adjacent one as its trace is nonzero.")
        else:
            pass
            #print("Error: generated matrix has zero-valued elements only and thus can't represent a graph.")


def generate_graph(adjacent_matrix):
    graph = nx.from_numpy_matrix(adjacent_matrix, parallel_edges=True)
    inc_mtx = nx.incidence_matrix(graph)
    inc_mtx_dense = scipy.sparse.csr_matrix.todense(inc_mtx)
    inc_mtx_dense_int = inc_mtx_dense.astype(int)
    nodes_list = graph.nodes()
    edges_list = graph.edges()
    nodes_data = pd.DataFrame(nodes_list)
    edges_data = pd.DataFrame(edges_list)
    return inc_mtx_dense_int, graph, nodes_data, edges_data


def generate_grid_graph(dim_A, dim_B):
    graph = nx.grid_graph(dim=(dim_A, dim_B))
    inc_mtx = nx.incidence_matrix(graph)
    inc_mtx_dense = scipy.sparse.csr_matrix.todense(inc_mtx)
    inc_mtx_dense_int = inc_mtx_dense.astype(int)
    nodes_list = graph.nodes()
    edges_list = graph.edges()
    nodes_data = pd.DataFrame(nodes_list)
    edges_data = pd.DataFrame(edges_list)
    return inc_mtx_dense_int, graph, nodes_data, edges_data


def generate_physical_values(dimension, source_value, incidence_matrix):
    edges_dim = np.shape(incidence_matrix)[1]
    incidence_T = incidence_matrix.transpose()
    incidence_T_inv = np.linalg.pinv(incidence_T)
    incidence_inv = np.linalg.pinv(incidence_matrix)
    conductivity_list = np.ones(edges_dim) + np.random.default_rng().uniform(-0.001, 0.001, edges_dim)  # ones + stochastic noise
    length_list = np.ones(edges_dim) + np.random.default_rng().uniform(-0.001, 0.001, edges_dim)        # vector from edges space
    source_list = np.zeros(dimension)             # vector from nodes space
    source_list[0] = source_value                 # but I want the central node to be the source, how to fix this?
    source_list[dimension-1] = -source_value      # but I want only outermost nodes to be the sinks, how to do this?
    # q = S * (delta^T)^-1
    flow_list = np.dot(source_list, incidence_T_inv)
    # delta*p = K/L * q
    pressure_diff_list = length_list * (1/conductivity_list) * flow_list
    pressure_list = np.dot(pressure_diff_list, incidence_inv)
    # x = delta^T * K/L *delta
    x = incidence_matrix  @ np.diag(1/length_list) @ np.diag(conductivity_list)  @ incidence_T
    x_dagger = np.linalg.pinv(x)  # Penrose pseudo-inverse

    # update data frames for both spaces
    if np.shape(nodes_data)[1] == 1:                                    # nodes are indexing by one int
        nodes_data.insert(loc=1, column='source', value=source_list)
        nodes_data.columns = ['nodes', 'source']
        nodes_data.insert(loc=2, column='pressure', value=pressure_list)
    elif np.shape(nodes_data)[1] == 2:                                  # nodes are indexing by two ints
        nodes_data.columns = ['no-', '-des']
        nodes_data['source'] = source_list
        nodes_data['pressure'] = pressure_list
    #print(nodes_data)
    edges_data.columns = ['ed-', '-ges']
    edges_data['length'] = length_list
    edges_data['conduct.'] = conductivity_list
    edges_data['flow'] = flow_list
    edges_data['press_diff'] = pressure_diff_list
    #print(edges_data)
    return x_dagger, incidence_T, incidence_inv, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list


def set_attributes(graph):
    colour_const = 2  # scaling constant to get element from colour-space from pressure-space
    # node_attrs = {tuple : dic, tuple: dic, ...} -- dic of (tuples as keys) and (dics as values)
    node_attrs = dict(graph.nodes)
    iterator = 0
    for key in node_attrs:
        vals = {"pressure": pressure_list[iterator], "colour": pressure_list[iterator] * colour_const}
        node_attrs[key] = vals
        iterator += 1
    nx.set_node_attributes(graph, node_attrs)
    #print(list(graph.nodes(data=True)))

    # now for edges
    colour_const = 3
    edge_attrs = dict(graph.edges)
    iterator = 0
    for key in edge_attrs:
        vals = {"length": length_list[iterator], "conductivity": conductivity_list[iterator], "flow": flow_list[iterator], "pressure_diff": pressure_diff_list[iterator],  "colour": flow_list[iterator] * colour_const}
        edge_attrs[key] = vals
        iterator += 1
    nx.set_edge_attributes(graph, edge_attrs)
    #print(list(graph.edges(data=True)))


def run_simulation_A(flow_list, conductivity_list, a, b, gamma, delta, flow_hat, c, r, dt, N):
    # solving diff eq
    t = 0
    for n in range(1, N+1):
        condition = any(k < 0 for k in conductivity_list) # in the future I'll just delete the edges with K == 0 -- their radius is zero
        if not False:
            iter = 0
            for e in graph.edges:
                con_val = conductivity_list[iter]
                #print(e, con_val)
                if con_val <= 0:
                    graph.remove_edge(*e)
                    print(e, 'removed')
                iter += 1
            t = dt * n
            # dK/dt = a*(exp(r*t/2)^(delta) * q / q_hat)^(2*gamma) - b * K + c
            # flow_list is separated into its sign and value, because I don't want to get complex values
            # the direction of the flow is maintained while its value changes
            # the information about direction is not coded into graph, so I have to be careful here
            dK = dt * (np.sign(flow_list) * np.float_power(a * (np.exp(r*t*delta/2) * np.abs(flow_list) / flow_hat), (2 * gamma)) - b * conductivity_list + c)
            conductivity_list += dK
            x = incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T
            x_dagger = np.linalg.pinv(x)
            # q = K/L * delta * (delta^T * K/L * delta)^dagger * S
            flow_list = source_list @ (x_dagger @ incidence_matrix) @ np.diag(conductivity_list) @ np.diag(1 / length_list)
            #pressure_diff_list = length_list * (1 / conductivity_list) * flow_list
            #pressure_list = np.dot(pressure_diff_list, incidence_inv)
    print('simulation time: ', t, ' seconds')

    # updating data frames
    edges_data['conduct.'] = conductivity_list
    edges_data['flow'] = flow_list
    #edges_data['press_diff'] = pressure_diff_list
    #print(edges_data)
    #nodes_data['pressure'] = pressure_list
    #print(nodes_data)


def graph_data_to_lists(graph):
    #edges
    conductivity_list = []
    flow_list = []
    length_list = []
    pressure_diff_list = []
    for i, f, data in graph.edges(data='conductivity'): conductivity_list.append(data)
    for i, f, data in graph.edges(data='flow'): flow_list.append(data)
    for i, f, data in graph.edges(data='length'): length_list.append(data)
    for i, f, data in graph.edges(data='pressure_diff'): pressure_diff_list.append(data)

    # nodes
    pressure_list = []
    if np.shape(nodes_data)[1] == 1:                                              # nodes are indexing by one int
        for i, data in graph.edges(data='pressure'): pressure_list.append(data)
    elif np.shape(nodes_data)[1] == 2:                                             # nodes are indexing by two ints
        for i, f, data in graph.edges(data='pressure'): pressure_list.append(data)

    #converting from list type to numpy arrays
    conductivity_list = np.asarray(conductivity_list)
    flow_list = np.asarray(flow_list)
    length_list = np.asarray(length_list)
    pressure_diff_list = np.asarray(pressure_diff_list)
    length_list = np.asarray(length_list)
    pressure_list = np.asarray(pressure_list)
    return conductivity_list, flow_list, length_list, pressure_diff_list, pressure_list


def run_simulation_G(graph, a, b, gamma, delta, flow_hat, c, r, dt, N):  # operating on data from graphs and updating data inside graphs
    # solving diff eq
    conductivity_list, flow_list, length_list, pressure_diff_list, pressure_list = graph_data_to_lists(graph)
    t = 0
    for n in range(1, N+1):
        condition = any(k < 0 for k in conductivity_list) # in the future I'll just delete the edges with K == 0 -- their radius is zero
        if not False:
            iterator = 0
            for e in graph.edges:
                con_val = graph.get_edge_data(*e)['conductivity']
                print(e, con_val)
                print(iterator, n)
                iterator += 1

                if con_val <= 0:
                    graph.remove_edge(*e)
                    print(e, 'removed')
            t = dt * n
            # dK/dt = a*(exp(r*t/2)^(delta) * q / q_hat)^(2*gamma) - b * K + c
            # flow_list is separated into its sign and value, because I don't want to get complex values
            # the direction of the flow is maintained while its value changes
            # the information about direction is not coded into graph, so I have to be careful here
            dK = dt * (np.sign(flow_list) * np.float_power(a * (np.exp(r*t*delta/2) * np.abs(flow_list) / flow_hat), (2 * gamma)) - b * conductivity_list + c)
            conductivity_list += dK
            x = incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T
            x_dagger = np.linalg.pinv(x)
            # q = K/L * delta * (delta^T * K/L * delta)^dagger * S
            flow_list = source_list @ (x_dagger @ incidence_matrix) @ np.diag(conductivity_list) @ np.diag(1 / length_list)
            #pressure_diff_list = length_list * (1 / conductivity_list) * flow_list
            #pressure_list = np.dot(pressure_diff_list, incidence_inv)
    print('simulation time: ', t, ' seconds')

    # updating data frames
    edges_data['conduct.'] = conductivity_list
    edges_data['flow'] = flow_list
    #edges_data['press_diff'] = pressure_diff_list
    #print(edges_data)
    #nodes_data['pressure'] = pressure_list
    #print(nodes_data)


def draw_graph(graph, name, pos):
    #straight_lines = nx.multipartite_layout(graph)    # sets nodes in straight lines
    nx.draw_networkx(graph, pos=pos)
    nx.draw_networkx_nodes(graph, pos=pos)
    nx.draw_networkx_edges(graph, pos=pos, width=conductivity_list*3)
    plt.savefig("%s.png" % name)
    # nodes colour - heatmap of pressure
    # edges length - proportional to length
    # edges colour - proportional to flow
    # edges arrows - in alignment with the sign of flow
    # edges thickness - proportional to (conductivity)^(-4)s
    # node_color=range(24), node_size=800, cmap=plt.cm.Blues


if __name__ == '__main__':
    start_time = time.time()

    n = 3
    number_of_nodes = n*n
    adjacency_matrix = generate_random_adjacent_matrix(number_of_nodes)
    #incidence_matrix, graph, nodes_data, edges_data = generate_graph(adjacency_matrix)  # random graph
    incidence_matrix, graph, nodes_data, edges_data = generate_grid_graph(n, n)          # regular grid
    source_value = 5
    x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list = generate_physical_values(number_of_nodes, source_value, incidence_matrix)
    set_attributes(graph)

    print(edges_data)
    pos = nx.spring_layout(graph)
    #pos_x = nx.multipartite_layout(graph, subset_key="layer")
    print(graph)
    draw_graph(graph, "graph", pos)
    # dK/dt = a*(exp(r*t/2)^(delta) * q / q_hat)^(2*gamma) - b * K + c
    parameters_set = {'a': 1.7, 'b': 0.5, 'gamma': 2/3, 'delta': 2.1, 'flow_hat': 3.4, 'c': 0, 'r': 1, 'dt': 0.001, 'N': 300}
    #run_simulation_A(flow_list, conductivity_list, **parameters_set)
    run_simulation_G(graph, **parameters_set)
    draw_graph(graph, "final_graph", pos)
    print(graph)
    print(edges_data)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))