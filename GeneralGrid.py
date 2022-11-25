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
    print(nodes_data)
    edges_data.columns = ['ed-', '-ges']
    edges_data['length'] = length_list
    edges_data['conduct.'] = conductivity_list
    edges_data['flow'] = flow_list
    edges_data['press_diff'] = pressure_diff_list
    print(edges_data)
    return x_dagger, incidence_T, incidence_inv, source_list, pressure_list, length_list, conductivity_list, flow_list


def run_simulation(flow_list, conductivity_list, a, b, gamma, delta, flow_hat, c, r, dt, N):
    # solving diff eq
    t = 0
    for n in range(1, N+1):
        if not any(k < 0 for k in conductivity_list):  # in the future I'll just delete the edges with K == 0 -- their radius is zero
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
            pressure_diff_list = length_list * (1 / conductivity_list) * flow_list
            pressure_list = np.dot(pressure_diff_list, incidence_inv)
    print('simulation time: ', t, ' seconds')

    # updating data frames
    edges_data['conduct.'] = conductivity_list
    edges_data['flow'] = flow_list
    edges_data['press_diff'] = pressure_diff_list
    print(edges_data)
    nodes_data['pressure'] = pressure_list
    print(nodes_data)


def draw_graph(graph):
    nx.draw_networkx(graph)
    plt.savefig("graph.png")
    # show nodes number
    # nodes colour - heatmap of pressure
    # edges length - proportional to length
    # edges thickness - proportional to (conductivity)^(-4)




if __name__ == '__main__':
    start_time = time.time()

    number_of_nodes = 6*6
    adjacency_matrix = generate_random_adjacent_matrix(number_of_nodes)
    #incidence_matrix, graph, nodes_data, edges_data = generate_graph(adjacency_matrix)  # random graph
    incidence_matrix, graph, nodes_data, edges_data = generate_grid_graph(6, 6)          # regular grid
    source_value = 5
    x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list = generate_physical_values(number_of_nodes, source_value, incidence_matrix)
    draw_graph(graph)
    parameters_set = {'a': 1.7, 'b': 0.5, 'gamma': 2/3, 'delta': 2.1, 'flow_hat': 3.4, 'c': 0.9, 'r': 1, 'dt': 0.01, 'N': 100}
    run_simulation(flow_list, conductivity_list, **parameters_set)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))