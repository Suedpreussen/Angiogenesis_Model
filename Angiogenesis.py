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
import sys
import matplotlib.animation as animation

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
run_simulation
draw_graph

unused code sent to Utils file
"""


def generate_grid_graph(dim_A, dim_B, hexagonal=False, triangular=False):
    if hexagonal:
        graph = nx.hexagonal_lattice_graph(dim_A, dim_B, periodic=False)
    elif triangular:
        graph = nx.triangular_lattice_graph(dim_A, dim_B, periodic=False)
    else:
        graph = nx.grid_graph(dim=(dim_A, dim_B))
    inc_mtx = nx.incidence_matrix(graph)
    inc_mtx_dense = scipy.sparse.csr_matrix.todense(inc_mtx)
    inc_mtx_dense_int = inc_mtx_dense.astype(int)
    #print(np.shape(inc_mtx_dense_int))
    #print(np.shape(nx.adjacency_matrix(graph)))
    nodes_list = graph.nodes()
    edges_list = graph.edges()
    nodes_data = pd.DataFrame(nodes_list)
    edges_data = pd.DataFrame(edges_list)
    return inc_mtx_dense_int, graph, nodes_data, edges_data


def generate_physical_values(graph, source_value, incidence_matrix, corridor_model=True, square=False):
    dimension = np.shape(incidence_matrix)[0]
    edges_dim = np.shape(incidence_matrix)[1]

    incidence_T = incidence_matrix.transpose()
    incidence_T_inv = np.linalg.pinv(incidence_T)
    incidence_inv = np.linalg.pinv(incidence_matrix)

    # checking moore-penrose inverse definition
    #print(np.allclose(incidence_T_inv @ incidence_T @ incidence_T_inv, incidence_T_inv))
    #print(np.allclose(incidence_T @ incidence_T_inv @ incidence_T, incidence_T))

    eps = 0.01
    conductivity_list = np.ones(edges_dim) + np.random.default_rng().uniform(-eps, eps, edges_dim)  # ones + stochastic noise
    length_list = np.ones(edges_dim) + np.random.default_rng().uniform(-eps, eps, edges_dim)        # vector from edges space


    # source in one node on the left and sink in one node on the right -- resulting in a corridor between them
    if corridor_model:
        source_list = np.zeros(dimension)             # vector from nodes space
        source_list[0] = source_value
        source_list[dimension-1] = -source_value
        #print("SOURCE", source_list)
    # source for square lattice -- eye retina model

    if square and dimension % 2 != 0:
        source_list = np.zeros(dimension)             # vector from nodes space
        source_list[int((dimension-1)/2)] = source_value             # source in the center

        iterator = 0
        for x, y in graph.nodes:        # accessing nodes on the border of the network
            if x == 0 or x == dimension-1:
                source_list[iterator] = -source_value/(np.sqrt(dimension))
        iterator += 1


    # q = (delta^T)^-1 * S
    flow_list = np.dot(source_list, incidence_T_inv)
    #print("FLOW", flow_list)

    # delta*p = K/L * q
    pressure_diff_list = flow_list * length_list * (1/conductivity_list)
    pressure_list = np.dot(pressure_diff_list, incidence_inv)

    # x = delta^T * K/L *delta
    x = incidence_matrix  @ np.diag(1/length_list) @ np.diag(conductivity_list) @ incidence_T


    x_dagger = np.linalg.pinv(x)  # Penrose pseudo-inverse

    return incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list


def update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time = False):
    # update data frames for both spaces
    if first_time:
        if np.shape(nodes_data)[1] == 1:                                    # nodes are indexing by one int
            nodes_data.columns = ['nodes']
            nodes_data['pressure'] = pressure_list
        elif np.shape(nodes_data)[1] == 2:                                  # nodes are indexing by two ints
            nodes_data.columns = ['no-', '-des']
            nodes_data['pressure'] = pressure_list
        #print(nodes_data)
        edges_data.columns = ['ed-', '-ges']
        edges_data['length'] = length_list
        edges_data['conduct.'] = conductivity_list
        edges_data['flow'] = flow_list
        edges_data['press_diff'] = pressure_diff_list
        #print(edges_data)
    else:
        # updating data frames
        edges_data['conduct.'] = conductivity_list
        edges_data['flow'] = flow_list
        edges_data['press_diff'] = pressure_diff_list
        #print(edges_data)
        nodes_data['pressure'] = pressure_list
        #print(nodes_data)



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
        vals = {"length": length_list[iterator], "conductivity": conductivity_list[iterator],
                "flow": flow_list[iterator], "pressure_diff": pressure_diff_list[iterator],
                "edge_colour": flow_list[iterator] * colour_const}
        edge_attrs[key] = vals
        iterator += 1
    nx.set_edge_attributes(graph, edge_attrs)
    #print(list(graph.edges(data=True)))


def checking_Murrays_law():
    # Q = const * r^3
    # r^3 = sum over cubes of radii of daughter branches
    # first of all, network needs to be hierarchical
    pass


def checking_Kirchhoffs_law(graph, source_list):
    index = 0
    successful_nodes = 0
    for node in graph.nodes(data=False):
        sum = 0
        for edge in graph.edges(node):
            sum += graph[edge[0]][edge[1]]['flow']
        if -1e-11 < sum - source_list[index] < 1e-11:
            #print("Kirchhoff's law at node {} fulfilled".format(node))
            successful_nodes += 1
        else:
            print("Kirchhoff's law at node {} NOT fulfilled!".format(node), sum - source_list[index])
        index += 1
    print(successful_nodes, graph.number_of_nodes())
    if successful_nodes == graph.number_of_nodes():
        print("SUCCESS! Kirchhoff's law fulfilled!")


def energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=False):
    # calculating energy functional E = sum over edges L * Q^2 / K
    energy_list = length_list * flow_list * flow_list / conductivity_list
    energy = np.sum(energy_list)

    # checking cost constraint = sum over edges L * K^(1/gamma - 1)
    constraint = np.sum(length_list * np.float_power(conductivity_list, (1/gamma - 1)))
    # what does it mean when it's not constant?

    if show_result:
        print("Energy: ", energy)
        print("Constraint: ", constraint)


def run_simulation(nodes_data, edges_data, x, x_dagger, incidence_T_inv, incidence_inv, incidence_T, incidence_matrix, graph, source_list,
                     pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list,
                     a, b, gamma, delta, nu, flow_hat, c, r, dt, N, is_scaled=False):
    # solving diff eq
    # exp(r*t/2)^(delta)
    # np.exp(r*t*delta/2)
    t = 0

    # implementing scaling factors
    if is_scaled:
        source_list = source_list * np.exp(r*t*delta/2)
        pressure_list = pressure_list * np.exp(r*t*nu/2)
        length_list = length_list * np.exp(r*t/2)
        conductivity_list = conductivity_list * np.exp(r*t*delta*gamma)
        flow_list = flow_list * np.exp(r*t*delta/2)

    energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)

    for n in range(1, N+1):
        t += dt

        # pruning implementation
        iter = 0
        for e in graph.edges:
            con_val = conductivity_list[iter]
            #print(e, con_val)
            if con_val < 1e-9:
                graph.remove_edge(*e)
                print(e, 'removed')
            iter += 1

        # dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c
        dK = dt * (np.float_power(a * (np.abs(flow_list) / flow_hat), (2 * gamma)) - b * conductivity_list + c * np.ones(len(flow_list)))
        conductivity_list += dK

        x = incidence_matrix @ np.diag(1/length_list) @ np.diag(conductivity_list) @ incidence_T

        x_dagger = np.linalg.pinv(incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T)

        # q = K/L * delta * (delta^T * K/L * delta)^dagger * S
        flow_list = source_list @ x_dagger @ incidence_matrix @ np.diag(conductivity_list) @ np.diag(1 / length_list)
        pressure_diff_list = length_list * (1 / conductivity_list) * flow_list
        pressure_list = np.dot(pressure_diff_list, incidence_inv)
        energy_functional(conductivity_list, length_list, flow_list, gamma)

        # updating data in graph dics
        set_attributes(graph, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list)

    energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)
    print('simulation time: ', t, ' seconds')
    update_df(pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)


def draw_graph(graph, name, pos, conductivity_list, flow_list, n):
    nx.draw_networkx(graph, pos=pos)
    nx.draw_networkx_nodes(graph, pos=pos, node_size=300/(2*n))
    nx.draw_networkx_edges(graph, pos=pos, width=conductivity_list) #, edge_color=flow_list   + np.ones(len(conductivity_list))  np.float_power(conductivity_list, 4)

    plt.axis('off')
    plt.savefig("%s.png" % name)
    #plt.show()
    # nodes colour - heatmap of pressure
    # edges length - proportional to length
    # edges colour - proportional to flow
    # edges arrows - in alignment with the sign of flow
    # edges thickness - proportional to (conductivity)^(-4)s
    # node_color=range(24), node_size=800, cmap=plt.cm.Blues
    """
    
    """