import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import matplotlib
import seaborn as sns
from bokeh.io import export_png, export_svgs
from bokeh.models import ColumnDataSource, DataTable, TableColumn

"""
General idea:
adjacency matrix --> incidence matrix, nodes, edges --> 
---> assign physical attributes --> run simulation --> create visualisation
"""

"""
TABLE OF CONTENT
--------------------

generate_random_adjacent_matrix  # not used
generate_graph  # not used
generate_grid_graph
generate_physical_values
update_df
set attributes
update_graph_data
run_simulation_A
graph_data_to_lists  # not used
run_simulation_G     # not used
draw_graph

"""


def generate_random_adjacent_matrix(dimension):   # dimension == #nodes
    # undirected graph
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
    return x_dagger, incidence_T, incidence_inv, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list


def update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time = False):
    # update data frames for both spaces
    if first_time:
        if np.shape(nodes_data)[1] == 1:                                    # nodes are indexing by one int
            #nodes_data.insert(loc=1, column='source', value=source_list)
            nodes_data.columns = ['nodes', 'source']
            nodes_data.insert(loc=2, column='pressure', value=pressure_list)
        elif np.shape(nodes_data)[1] == 2:                                  # nodes are indexing by two ints
            nodes_data.columns = ['no-', '-des']
            nodes_data['pressure'] = pressure_list
        print(nodes_data)
        edges_data.columns = ['ed-', '-ges']
        edges_data['length'] = length_list
        edges_data['conduct.'] = conductivity_list
        edges_data['flow'] = flow_list
        edges_data['press_diff'] = pressure_diff_list
        print(edges_data)
    else:
        # updating data frames
        edges_data['conduct.'] = conductivity_list
        edges_data['flow'] = flow_list
        edges_data['press_diff'] = pressure_diff_list
        print(edges_data)
        nodes_data['pressure'] = pressure_list
        print(nodes_data)


def set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list):
    colour_const = 2  # scaling constant to get element from colour-space from pressure-space
    # node_attrs = {tuple : dic, tuple: dic, ...} -- dic of (tuples as keys) and (dics as values)
    node_attrs = dict(graph.nodes)
    iterator = 0
    for key in node_attrs:
        vals = {"pressure": pressure_list[iterator], "node_colour": pressure_list[iterator] * colour_const}
        node_attrs[key] = vals
        iterator += 1
    nx.set_node_attributes(graph, node_attrs)
    #print(list(graph.nodes(data=True)))

    # now for edges
    colour_const = 3
    edge_attrs = dict(graph.edges)
    iterator = 0
    for key in edge_attrs:
        vals = {"length": length_list[iterator], "conductivity": conductivity_list[iterator], "flow": flow_list[iterator], "pressure_diff": pressure_diff_list[iterator],  "edge_colour": flow_list[iterator] * colour_const}
        edge_attrs[key] = vals
        iterator += 1
    nx.set_edge_attributes(graph, edge_attrs)
    #print(list(graph.edges(data=True)))


def update_graph_data(graph):
    pass
    # iterate over attributes and update values
    # node_attrs = {tuple : dic, tuple: dic, ...} -- dic of (tuples as keys) and (dics as values)
    for tuples_as_keys in graph.nodes(data=True):
        for dics_as_keys in graph.nodes(data="pressure"):
            pass
        for dics_as_keys in graph.nodes(data="node_colour"):
            pass


def checking_Murrays_law():
    # Q = const * r^3
    # r^3 = sum over cubes of radii of daughter branches
    pass


def checking_Kirchhoffs_law():
    # 100 * |theoretical value - simulated value| / th. val. = diff. between theory and simulation in percents
    # sum of flows = sum of sources = 0
    pass


def run_simulation_A(nodes_data, edges_data, incidence_inv, incidence_T, incidence_matrix, graph, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, a, b, gamma, delta, flow_hat, c, r, dt, N):
    # solving diff eq
    # without scaling factors
    t = 0
    for n in range(1, N+1):

        # pruning implementation
        iter = 0
        for e in graph.edges:
            con_val = conductivity_list[iter]
            #print(e, con_val)
            if con_val <= 0:
                graph.remove_edge(*e)
                print(e, 'removed')
            iter += 1

        t = dt * n
        # dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c                                                                #exp(r*t/2)^(delta) *
        dK = dt * (np.float_power(a * (np.abs(flow_list) / flow_hat), (2 * gamma)) - b * conductivity_list + c)      # np.exp(r*t*delta/2) *
        conductivity_list += dK
        x = incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T
        x_dagger = np.linalg.pinv(x)
        # q = K/L * delta * (delta^T * K/L * delta)^dagger * S
        flow_list = source_list @ (x_dagger @ incidence_matrix) @ np.diag(conductivity_list) @ np.diag(1 / length_list)
        pressure_diff_list = length_list * (1 / conductivity_list) * flow_list   # 1/conduct generates infinities when conduct approaches zero!!
        pressure_list = np.dot(pressure_diff_list, incidence_inv)

        # updating data in graph dics
        set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list)

    print('simulation time: ', t, ' seconds')
    update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)

"""
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
    if np.shape(pd.DataFrame(graph.nodes))[1] == 1:                                              # nodes are indexing by one int
        for i, data in graph.edges(data='pressure'): pressure_list.append(data)
    elif np.shape(pd.DataFrame(graph.nodes))[1] == 2:                                             # nodes are indexing by two ints
        for i, f, data in graph.edges(data='pressure'): pressure_list.append(data)

    #converting from list type to numpy arrays
    conductivity_list = np.asarray(conductivity_list)
    flow_list = np.asarray(flow_list)
    length_list = np.asarray(length_list)
    pressure_diff_list = np.asarray(pressure_diff_list)
    length_list = np.asarray(length_list)
    pressure_list = np.asarray(pressure_list)
    return conductivity_list, flow_list, length_list, pressure_diff_list, pressure_list
"""

"""
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
"""


def draw_graph(graph, name, pos, conductivity_list):
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

"""
def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "white"],
              [norm(1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)         # this code could be useful
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.g
    fig.savefig(path)
"""