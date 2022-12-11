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


def generate_grid_graph(dim_A, dim_B, periodic=False, hexagonal=False, triangular=False):
    if hexagonal:
        graph = nx.hexagonal_lattice_graph(dim_A, dim_B, periodic=periodic)
    elif triangular:
        graph = nx.triangular_lattice_graph(dim_A, dim_B, periodic=periodic)
    else:
        graph = nx.grid_graph(dim=(dim_A, dim_B))
    inc_mtx = nx.incidence_matrix(graph)
    inc_mtx_dense = scipy.sparse.csr_matrix.todense(inc_mtx)
    inc_mtx_dense_int = inc_mtx_dense.astype(int)

    # change inc_mtx to a directed one
    for row in inc_mtx_dense_int.T:
        ones_count = 0
        row_element_count = 0
        for element in row:
            if element == 1:
                ones_count += 1
                #print("el", element)
                #print("c", ones_count)
                if ones_count == 2:
                    row[row_element_count] *= -1
                    #print("asas")
            row_element_count +=1

    #print(np.shape(inc_mtx_dense_int))
    #print(np.shape(nx.adjacency_matrix(graph)))
    nodes_list = graph.nodes()
    edges_list = graph.edges()
    nodes_data = pd.DataFrame(nodes_list)
    edges_data = pd.DataFrame(edges_list)
    return inc_mtx_dense_int, graph, nodes_data, edges_data


def generate_physical_values(graph, source_value, incidence_matrix, corridor_model=False, two_capacitor_plates_model=False,
                             one_capacitor_plates_model=False, three_sides_model=False, quater_model=False, square_concentric_model=False, triangular=False):
    dimension = np.shape(incidence_matrix)[0]
    edges_dim = np.shape(incidence_matrix)[1]

    incidence_T = incidence_matrix.transpose()
    incidence_T_inv = np.linalg.pinv(incidence_T)
    incidence_inv = np.linalg.pinv(incidence_matrix)
    #print(incidence_matrix)
    # checking moore-penrose inverse definition
    #print(np.allclose(incidence_T_inv @ incidence_T @ incidence_T_inv, incidence_T_inv))
    #print(np.allclose(incidence_T @ incidence_T_inv @ incidence_T, incidence_T))

    eps = 0.9
    conductivity_list = np.ones(edges_dim) + np.random.default_rng().uniform(-eps, eps, edges_dim)  # ones + stochastic noise
    length_list = np.ones(edges_dim) # + np.random.default_rng().uniform(-eps, eps, edges_dim)        # vector from edges space


    # source in one node on the left and sink in one node on the right -- resulting in a corridor between them
    if corridor_model:
        source_list = np.zeros(dimension)             # vector from nodes space
        source_list[0] = source_value
        source_list[dimension-1] = -source_value
        #print(dimension)
        #print(int(np.sqrt(dimension)/2)-1)
        #print(dimension-int(np.sqrt(dimension)/2))
        #print("SOURCE", source_list)

    if two_capacitor_plates_model:
        source_list = np.zeros(dimension)             # vector from nodes space
        last_index = int(np.sqrt(dimension)-1)
        nodes_on_one_side = int(np.sqrt(dimension))
        iterator = 0
        for iterator in range(0, last_index+1):
            source_list[iterator] = source_value/nodes_on_one_side
            source_list[dimension-1-iterator] = -source_value/nodes_on_one_side
            iterator += 1
        print(dimension)
        print(last_index)
        print(nodes_on_one_side)

    if one_capacitor_plates_model:
        source_list = np.zeros(dimension)             # vector from nodes space
        last_index = int(np.sqrt(dimension)-1)
        nodes_on_one_side = int(np.sqrt(dimension)/2)
        source_list[int(np.sqrt(dimension) / 2) - 1] = source_value
        iterator = 0
        for iterator in range(0, last_index+1, 2):
            print(iterator)
            source_list[dimension-2-iterator] = -source_value/nodes_on_one_side
            iterator += 1
        #print(source_list)
        print(dimension)
        print(last_index)
        print(nodes_on_one_side)

    # source for square lattice -- eye retina model
    # works only for odd number of rows/columns -- only then a central node exists
    if square_concentric_model:
        source_list = np.zeros(dimension)                            # vector from nodes space
        source_list[int((dimension-1)/2)] = source_value             # source in the center
        number_of_boundary_nodes = 4*np.sqrt(dimension)-4
        last_index = int(np.sqrt(dimension)-1)
        iterator = 0
        for node in graph.nodes:        # accessing nodes on the boundaries of the network
            if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                source_list[iterator] = -source_value/number_of_boundary_nodes
                #print(node)
            iterator += 1
        print(number_of_boundary_nodes)
        print(-source_value / (4*np.sqrt(dimension)-4) * iterator)
        print(source_value)


    if triangular:
        source_list = np.zeros(dimension)
        source_list[int((dimension - 1) / 2)] = source_value
        number_of_bordering_nodes = 16
        last_index = int(np.sqrt(dimension) - 1)
        iterator = 0
        for node in graph.nodes:  # accessing nodes on the border of the network
            if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                source_list[iterator] = source_value / number_of_bordering_nodes  # number of nodes on the border
                print(node)
            iterator += 1

    """
    if side_to_side:
        source_list = np.zeros(dimension)
        last_index = int(np.sqrt(dimension)-1)
        nodes_on_one_side = int(np.sqrt(dimension))
        iterator = 0
        for node in graph.nodes:
            if node[0] == 0:    # in-flow side
                source_list[iterator] = source_value / (2*nodes_on_one_side)
                print("source", node)
            elif node[0] == last_index:   # out-flow side
                source_list[iterator] = -source_value / (2*nodes_on_one_side)
                print("sink", node)

            iterator += 1
    """


    # q = (delta^T)^-1 * S
    #print(incidence_T_inv)
    flow_list = np.dot(source_list, incidence_T_inv)

    #print("FLOW", flow_list)

    # delta*p = K/L * q
    pressure_diff_list = flow_list * length_list * (1/conductivity_list)
    pressure_list = np.dot(pressure_diff_list, incidence_inv)

    # x = delta^T * K/L *delta
    x = incidence_matrix  @ np.diag(1/length_list) @ np.diag(conductivity_list) @ incidence_T
    x_dagger = np.linalg.pinv(x)  # Penrose pseudo-inverse

    return incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list


def update_df(source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time = False):
    # update data frames for both spaces
    if first_time:
        if np.shape(nodes_data)[1] == 1:                                    # nodes are indexing by one int
            nodes_data.columns = ['nodes']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
        elif np.shape(nodes_data)[1] == 2:                                  # nodes are indexing by two ints
            nodes_data.columns = ['no-', '-des']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
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
        nodes_data['source'] = source_list
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


def checking_Kirchhoffs_law(graph, source_list, flow_list):
    #print(source_list)
    index = 0
    successful_nodes = 0
    for node in graph.nodes(data=False):
        sum = 0
        #print(node, "___________")
        for edge in graph.edges(node):
            if np.sum(edge[0]) < np.sum(edge[1]):        # implementing direction of flow to the undirected graph
                sum += graph[edge[0]][edge[1]]['flow']
                #print(edge, '|',  np.sum(edge[0]), '|',  np.sum(edge[1]), '|',  graph.get_edge_data(*edge)['flow'])
            else:
                sum -= graph[edge[0]][edge[1]]['flow']
                #print("ELSE", edge, '|',  np.sum(edge[0]), '|',  np.sum(edge[1]), '|',  -graph.get_edge_data(*edge)['flow'])
        if -1e-11 < sum - source_list[index] < 1e-11:
            #print("Kirchhoff's law at node {} fulfilled".format(node))
            #print(sum, '    |', source_list[index], '    |', print(node))
            successful_nodes += 1
        else:
            pass
            print(sum, '|', source_list[index], '|', print(node))
            #print("Kirchhoff's law at node {} NOT fulfilled!".format(node), sum + source_list[index])
        index += 1
    print("number of nodes fulfilling K's law:", successful_nodes, 'out of', graph.number_of_nodes())
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


def run_simulation(source_value, m, pos, nodes_data, edges_data, x, x_dagger, incidence_T_inv, incidence_inv, incidence_T, incidence_matrix, graph, source_list,
                     pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list,
                     a, b, gamma, delta, nu, flow_hat, c, r, dt, N, is_scaled=False, with_pruning=False):
    # solving diff eq
    # exp(r*t/2)^(delta)
    # np.exp(r*t*delta/2)
    t = 0



    # two control parameters
    rho = b/(r*gamma*delta)   # the ratio between the time scales for adaptation and growth
    print("RHO: ", rho)

    source_hat = source_value
    kappa = (c/a)*np.float_power((flow_hat/source_hat), 2*gamma)  # a/c is the ratio between background growth rate and adaptation strength and the hatted quantities are typical scales for flow and source strength.
    print("KAPPA: ", kappa)


    # implementing scaling factors
    if is_scaled:
        source_list = source_list * np.exp(r*t*delta/2)
        pressure_list = pressure_list * np.exp(r*t*nu/2)
        length_list = length_list * np.exp(r*t/2)
        conductivity_list = conductivity_list * np.exp(r*t*delta*gamma)
        flow_list = flow_list * np.exp(r*t*delta/2)
        b = b + r*gamma*delta
        c = c * np.exp(-r*t*gamma*delta)

    energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)
    number_of_removed_edges = 0
    number_of_removed_nodes = 0

    lagrange_multiplier = 0.000001
    flow_from_lagrange_optimisation = np.sqrt(lagrange_multiplier)*np.sqrt(1/gamma +1)*np.float_power(conductivity_list, 1/(2*gamma))
    for n in range(1, N+1):
        t += dt
        if n!= 0 and n!= N and n == N/2:
            draw_graph(graph, "mid_graph", pos, conductivity_list, m)

        # pruning implementation
        if with_pruning:
            for edge in graph.edges:
                con_val = graph.get_edge_data(*edge)["conductivity"]
                #print(*edge, " | ", con_val)
                if con_val < 1e-11:                 # removing an edge if its radius is ~= 0
                    graph.remove_edge(*edge)
                    print('edge', *edge, 'removed')
                    number_of_removed_edges += 1

            for node in dict(graph.nodes).copy():           # for reasons unknown I had to cast graph.nodes into dict
                                                            # and use dictionary method copy() to get away with error
                                                            # "RuntimeError: dictionary changed size during iteration"
                                                            # but with edges it worked just fine
                neighbors = set(nx.neighbors(graph, node))
                #print(len(neighbors))
                if len(neighbors) == 0:
                    graph.remove_node(node)
                    print('node', node, 'removed')
                    #pos = dict((m, m) for m in graph.nodes())
                    number_of_removed_nodes += 1


        # dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c
        dK = dt * (np.float_power(a * (np.abs(flow_list) / flow_hat), (2 * gamma)) - b * conductivity_list + c * np.ones(len(flow_list)))
        #dK = dt * (np.float_power(a * (np.abs(flow_from_lagrange_optimisation) / flow_hat), (2 * gamma)) - b * conductivity_list + c * np.ones(len(flow_list)))
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

        #dEdF_lam_dgdF = np.sum(flow_list) / np.sum(conductivity_list)    # checking first eq from lagrange optimisation
        #print(dEdF_lam_dgdF)


    energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)
    print('simulation time: ', t, ' seconds')
    print("number of removed edges: ", number_of_removed_edges)
    print("number of removed nodes: ", number_of_removed_nodes)

    update_df(source_list, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)
    return graph, conductivity_list


def draw_graph(graph, name, pos, conductivity_list, n):
    if len(conductivity_list) < 100:
        labels = nx.get_edge_attributes(graph, 'flow')
        for key, val in labels.items():
            new_val = round(val, 2)
            labels[key] = new_val
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, rotate=False, font_color='red')
        nx.draw_networkx(graph, pos=pos, node_size=400/(n))
        nx.draw_networkx_nodes(graph, pos=pos, node_size=400/(n))
        nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1/4)*2)  #, edge_color=flow_list   + np.ones(len(conductivity_list))  np.float_power(conductivity_list, 4)
    else:
        #nx.draw_networkx(graph, pos=pos)
        nx.draw_networkx_nodes(graph, pos=pos, node_size=200 / (2 * n))
        nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1/4)*2)

    plt.axis('off')
    plt.axis('scaled')
    plt.savefig("%s.png" % name)
    plt.clf()
    #plt.show()
    # nodes colour - heatmap of pressure
    # edges length - proportional to length
    # edges colour - proportional to flow
    # edges arrows - in alignment with the sign of flow
    # edges thickness - proportional to (conductivity)^(-4)s
    # node_color=range(24), node_size=800, cmap=plt.cm.Blues
    """
    
    """