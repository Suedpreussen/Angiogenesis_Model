import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib as mpl

"""
General idea:
create grid --> assign physical attributes --> run simulation and create visualisation
"""

"""
TABLE OF CONTENT
--------------------
generate_grid_graph
localise_source
generate_physical_values
update_df
set_graph_attributes
update_graph_data
checking_Kirchhoffs_and_Murrays_law
draw_graph
draw_histogram
run_simulation

unused code sent to Utils file
"""


class Model:
    """Main class"""
    def __int__(self, number_of_rows_or_columns: int, shape_of_boundaries="a_square", type_of_lattice="square"):
        """Class constructor"""
        if type_of_lattice == "square":
            assert number_of_rows_or_columns % 2 != 0, "Square model needs to have an odd number of rows or columns to get a central node."

        # save given model parameters
        self.__number_of_rows_or_columns = number_of_rows_or_columns
        self.__shape_of_boundaries = shape_of_boundaries
        self.__type_of_lattice = type_of_lattice

    def __generate_lattice(self):
        """Setting up lattice"""
        pass

    def __localise_source(self):
        """Compute source vector"""
        pass

    def __compute_physical_values(self):
        pass

    def __update_pandas_data(self):
        pass

    def __update_networkx_data(self):
        pass

    def __check_kirchhoffs(self):
        pass

    def __compute_energy_dissipation(self):
        pass

    def draw_histogram(self):
        pass

    def draw_graph(self):
        pass

    def create_experiment_log(self):
        pass

    def run_simulation(self):
        pass



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
                if ones_count == 2:
                    row[row_element_count] *= -1
            row_element_count +=1

    nodes_list = graph.nodes()
    edges_list = graph.edges()
    nodes_data = pd.DataFrame(nodes_list)
    edges_data = pd.DataFrame(edges_list)
    return inc_mtx_dense_int, graph, nodes_data, edges_data


def localise_source(graph, source_value, corridor_model=False, two_capacitor_plates_model=False,
                             one_capacitor_plates_model=False, square_concentric_model=False,
                                veins_square_concentric_model=False, triangular=False):
    nodes_dim = graph.number_of_nodes()

    # source in one node on the left and sink in one node on the right -- resulting in a corridor between them
    if corridor_model:
        source_list = np.zeros(nodes_dim)
        source_list[0] = source_value
        source_list[nodes_dim-1] = -source_value

    if two_capacitor_plates_model:
        source_list = np.zeros(nodes_dim)
        last_index = int(np.sqrt(nodes_dim)-1)
        nodes_on_one_side = int(np.sqrt(nodes_dim))
        iterator = 0
        for iterator in range(0, last_index+1):
            source_list[iterator] = source_value/nodes_on_one_side
            source_list[nodes_dim-1-iterator] = -source_value/nodes_on_one_side
            iterator += 1

    if one_capacitor_plates_model:
        source_list = np.zeros(nodes_dim)
        last_index = int(np.sqrt(nodes_dim)-1)
        nodes_on_one_side = int(np.sqrt(nodes_dim)/2)
        source_list[int(np.sqrt(nodes_dim) / 2) - 1] = source_value
        iterator = 0
        for iterator in range(0, last_index+1, 2):
            print(iterator)
            source_list[nodes_dim-2-iterator] = -source_value/nodes_on_one_side
            iterator += 1

    # source in the center of the square lattice -- eye retina model
    # works only for odd number of rows/columns -- only then a central node exists
    if square_concentric_model:
        source_list = np.zeros(nodes_dim)
        source_list[int((nodes_dim-1)/2)] = source_value             # source in the center
        number_of_boundary_nodes = 4*np.sqrt(nodes_dim)-4
        #number_of_boundary_nodes = 4*np.sqrt(dimension)-4-4          # without edges on outermost ring
        last_index = int(np.sqrt(nodes_dim)-1)
        iterator = 0
        for node in graph.nodes:        # accessing nodes on the boundaries of the network
            if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                source_list[iterator] = -source_value/number_of_boundary_nodes
            iterator += 1

    # inverse of sources and sinks from the square concentric model
    if veins_square_concentric_model:
        source_list = np.zeros(nodes_dim)
        source_list[int((nodes_dim-1)/2)] = -source_value             # source in the center
        number_of_boundary_nodes = 4*np.sqrt(nodes_dim)-4
        last_index = int(np.sqrt(nodes_dim)-1)
        iterator = 0
        for node in graph.nodes:        # accessing nodes on the boundaries of the network
            if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                source_list[iterator] = source_value/number_of_boundary_nodes
                #print(node)
            iterator += 1

    # triangular grid model
    if triangular:
        source_list = np.zeros(nodes_dim)
        source_list[int((nodes_dim - 1) / 2)] = source_value
        number_of_bordering_nodes = 16
        last_index = int(np.sqrt(nodes_dim) - 1)
        iterator = 0
        for node in graph.nodes:  # accessing nodes on the border of the network
            if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                source_list[iterator] = source_value / number_of_bordering_nodes  # number of nodes on the border
            iterator += 1

    return source_list


def generate_physical_values(source_list, incidence_matrix):
    # nodes-space vectors: source, pressure
    # edges-space vectors: conductivity, length, flow, pressure difference
    nodes_dim = np.shape(incidence_matrix)[0]
    edges_dim = np.shape(incidence_matrix)[1]

    incidence_T = incidence_matrix.transpose()
    incidence_T_inv = np.linalg.pinv(incidence_T)
    incidence_inv = np.linalg.pinv(incidence_matrix)

    epsilon = 0.8
    radii_list = np.ones(edges_dim) + np.random.default_rng().uniform(-epsilon, epsilon, edges_dim)  # ones + stochastic noise
    conductivity_list = 0.3*np.float_power(radii_list, 4)
    length_list = 0.8*np.ones(edges_dim)

    # Q = (delta^T)^-1 * S
    flow_list = np.dot(source_list, incidence_T_inv)

    # delta_p = K/L * Q
    pressure_diff_list = flow_list * length_list * (1/conductivity_list)
    pressure_list = np.dot(pressure_diff_list, incidence_inv)

    # x = delta^T * K/L *delta
    x = incidence_matrix  @ np.diag(1/length_list) @ np.diag(conductivity_list) @ incidence_T
    x_dagger = np.linalg.pinv(x)  # Penrose pseudo-inverse

    # Q = K/L * delta * (delta^T * K/L * delta)^dagger * S
    flow_list = source_list @ x_dagger @ incidence_matrix @ np.diag(conductivity_list) @ np.diag(1 / length_list)

    return incidence_T_inv, x, x_dagger, incidence_inv, incidence_T, pressure_list, length_list, conductivity_list, flow_list, pressure_diff_list


def update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data, first_time= False):
    # creating data frames
    if first_time:
        if np.shape(nodes_data)[1] == 1:                         # if nodes are indexing by one int
            nodes_data.columns = ['nodes']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
        elif np.shape(nodes_data)[1] == 2:                       # if nodes are indexing by two ints
            nodes_data.columns = ['no-', '-des']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
        edges_data.columns = ['ed-', '-ges']
        edges_data['conductivity'] = conductivity_list
        edges_data['flow'] = np.abs(flow_list)
        edges_data['press_diff'] = pressure_diff_list
    # updating data frames
    else:
        edges_data['conductivity'] = conductivity_list
        edges_data['flow'] = np.abs(flow_list)
        edges_data['press_diff'] = pressure_diff_list
        nodes_data['pressure'] = pressure_list
        nodes_data['source'] = source_list


def set_graph_attributes(graph, pressure_list, conductivity_list, flow_list, pressure_diff_list):
    # node_attrs = {tuple : dic, tuple: dic, ...} -- dic of (tuples as keys) and (dics as values)
    node_attrs = dict(graph.nodes)
    iterator = 0
    for key in node_attrs:
        vals = {"pressure": pressure_list[iterator]}
        node_attrs[key] = vals
        iterator += 1
    nx.set_node_attributes(graph, node_attrs)
    # now for edges
    edge_attrs = dict(graph.edges)
    iterator = 0
    for key in edge_attrs:
        vals = {"conductivity": conductivity_list[iterator],
                "flow": flow_list[iterator], "pressure_diff": pressure_diff_list[iterator]}
        edge_attrs[key] = vals
        iterator += 1
    nx.set_edge_attributes(graph, edge_attrs)


def checking_Murrays_law(graph):
    # Q = const * r^alpha
    # K = const' * r^4  =>  r = const" * K^1/4
    # Q = const''' * K^alpha/4  =>  Q * K^-alpha/4 = const'''
    alpha = 7/3
    constant_list = []
    for edge in graph.edges():
        constant = np.abs(graph[edge[0]][edge[1]]['flow']) * np.float_power(graph[edge[0]][edge[1]]['conductivity'], -alpha/4)
        constant_list.append(constant)
    print(f"power index={alpha}:  ", constant_list)
    pass


def checking_Kirchhoffs_and_Murrays_law(graph, source_list):
    index = 0
    successful_Kirchhoffs_nodes = 0
    successful_Murrays_nodes = 0
    alpha = 7/3
    for node in graph.nodes(data=False):
        flow_sum = 0
        radii_in_sum = 0
        radii_out_sum = 0
        for edge in graph.edges(node):              # implementing direction of flow to the undirected graph
            if np.sum(edge[0]) < np.sum(edge[1]):
                flow_sum += graph[edge[0]][edge[1]]['flow']
                radii_in_sum += np.float_power(graph[edge[0]][edge[1]]['conductivity'], -alpha / 4)
            else:
                flow_sum -= graph[edge[0]][edge[1]]['flow']
                radii_out_sum -= np.float_power(graph[edge[0]][edge[1]]['conductivity'], -alpha / 4)

        if -1e-11 < flow_sum - source_list[index] < 1e-11:       # checking for every node if the sum of inflows and ouflows yields zero
            successful_Kirchhoffs_nodes += 1
        else:
            pass
            print(flow_sum, '|', source_list[index], '|', print(node))
            #print("Kirchhoff's law at node {} NOT fulfilled!".format(node), flow_sum + source_list[index])

        if -1e-11 < np.abs(radii_in_sum - radii_out_sum) < 1e-11:       # checking M's law
            successful_Murrays_nodes += 1
        else:
            pass
            #print(np.abs(radii_in_sum - radii_out_sum) , '||', print(node))
        index += 1

    print("number of nodes fulfilling K's law:", successful_Kirchhoffs_nodes, 'out of', graph.number_of_nodes())
    print("number of nodes fulfilling Murray's law:", successful_Murrays_nodes, 'out of', graph.number_of_nodes())

    if successful_Kirchhoffs_nodes == graph.number_of_nodes():
        print("SUCCESS! Kirchhoff's law fulfilled!")


def energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=False):
    # calculating energy functional E = sum over edges L * Q^2 / K
    energy_list = length_list * flow_list * flow_list / conductivity_list
    energy = np.sum(energy_list)

    # checking cost constraint = sum over edges L * K^(1/gamma - 1)
    constraint = np.sum(length_list * np.float_power(conductivity_list, (1/gamma - 1)))

    if show_result:
        print("Energy: ", energy)
        print("Constraint: ", constraint)


def draw_histogram(directory_name, edges_data, file_name):
    fig, ax = plt.subplots(1, 2)   #, figsize=(6.4, 4.8)
    # titles
    ax[0].set_title('Conductivity')
    ax[1].set_title('Flow')
    # draw histograms
    ax[0].hist(edges_data['conductivity'], bins=40)
    ax[1].hist(edges_data['flow'], bins=40)
    plt.savefig(f'{directory_name}/{file_name}.png', dpi=300)
    plt.clf()


def draw_global_histogram(directory_name, list_of_dfs):
    fig, ax = plt.subplots(1, 2)   #, figsize=(6.4, 4.8)
    # titles
    ax[0].set_title('Conductivity')
    ax[1].set_title('Flow')
    for df in list_of_dfs:
        # draw histograms
        ax[0].hist(df['conductivity'], bins=40)
        ax[1].hist(df['flow'], bins=40)
    plt.savefig(f"{directory_name}/global_histogram.png")
    plt.clf()


def draw_graph(directory_name, graph, name, conductivity_list, number_of_rowscols, triangular=False, hexagonal=False):
    max = 26
    cmap = plt.cm.magma_r
    # setting the layout for the graph visualisation
    if hexagonal:
        pos = nx.get_node_attributes(graph, 'pos')  # hexagonal rigid layout
    elif triangular:
        pos = nx.get_node_attributes(graph, 'pos')  # triangular rigid layout
    else:
        pos = dict((number_of_rowscols, number_of_rowscols) for number_of_rowscols in graph.nodes())  # square rigid layout

    # plot differently for different sizes of the network
    if len(conductivity_list) < 100:
        labels = nx.get_edge_attributes(graph, 'flow')
        for key, val in labels.items():
            new_val = round(val, 2)
            labels[key] = new_val
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, rotate=False, font_color='red')
        nx.draw_networkx(graph, pos=pos, node_size=400 / (number_of_rowscols))
        nx.draw_networkx_nodes(graph, pos=pos, node_size=400 / (number_of_rowscols))
        nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1 / 4) * 2)
    elif 99 < len(conductivity_list) < 400:
        nx.draw_networkx_nodes(graph, pos=pos, node_size=200 / (2 * number_of_rowscols))
        nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1 / 4) * 2,
                               edge_color=conductivity_list, edge_cmap=cmap, edge_vmin=0, edge_vmax=max)
    elif 399 < len(conductivity_list):
        # nx.draw_networkx_nodes(graph, nodelist=(n-1, n-1), pos=pos, node_size=100 / (2 * n), node_color='black')
        nc = nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1 / 4) * 1.5,
                                    edge_color=conductivity_list, edge_cmap=cmap, edge_vmin=0, edge_vmax=max)
    plt.axis('off')
    plt.axis('scaled')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)
    norm = mpl.colors.Normalize(vmin=0, vmax=max)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig(f"{directory_name}/{name}.png", bbox_inches=0, dpi=300)
    plt.clf()
    # plt.show()


def run_simulation(directory_name, source_value, number_of_rowscols, nodes_data, edges_data, incidence_inv,
                   incidence_T, incidence_matrix, graph, source_list,pressure_list, length_list,
                   conductivity_list, flow_list, pressure_diff_list,a, b, gamma, delta, nu,
                   flow_hat, c, r, dt, N, is_scaled=False, with_pruning=False):
    t = 0
    checking_Kirchhoffs_and_Murrays_law(graph, source_list)

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
        print("time unit: ", 1/b)

    lagrange_multiplier = 0.01
    flow_from_lagrange_optimisation = np.sqrt(lagrange_multiplier)*np.sqrt(1/gamma+1)*np.float_power(conductivity_list, 1/(2*gamma))

    # snapshot before the sim
    # draw graphs
    update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)
    draw_graph(directory_name,graph, f"graph_at_0_{N}", conductivity_list, number_of_rowscols)
    draw_histogram(directory_name, edges_data, f"histogram_0_{N}")
    # print log
    print(f"______n = 0________")
    print("Q_av: ", np.average(np.abs(flow_list)))
    energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)
    print("Sum of conductivity: ", np.sum(conductivity_list))

    list_of_dfs = []                     # container to store dfs at snapshots
    list_of_dfs.append(edges_data)

    # MAIN LOOP
    for n in range(1, N+1):
        t += dt

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
        set_graph_attributes(graph, pressure_list, conductivity_list, flow_list, pressure_diff_list)

        # sim snapshots
        if n == N or n == N/16 or n == (2*N)/16 or n == (3*N)/16 or n == N/4 or n == N/2 or n == (3*N)/4 or n == N/32 or n == (2*N)/32 or n == (3*N)/N:
            # draw graphs
            update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)
            draw_graph(directory_name, graph, f"graph_at_{n}_{N}", conductivity_list, number_of_rowscols)
            list_of_dfs.append(edges_data)
            draw_histogram(directory_name, edges_data, f"histogram_{n}_{N}")
            # print log
            print(f"________n = {n}________")
            print("Q_av: ", np.average(np.abs(flow_list)))
            energy_functional(conductivity_list, length_list, flow_list, gamma, show_result=True)
            print("Sum of conductivity: ", np.sum(conductivity_list))

        #dEdF_lam_dgdF = np.sum(flow_list) / np.sum(conductivity_list)    # checking first eq from lagrange optimisation
        #print(dEdF_lam_dgdF)

    print('simulation time: ', round(t*b, 3), "1/(b')  =  ", round(t, 3), "seconds")
    update_df(source_list, pressure_list, conductivity_list, flow_list, pressure_diff_list, nodes_data, edges_data)
    checking_Kirchhoffs_and_Murrays_law(graph, source_list)
    draw_global_histogram(directory_name, list_of_dfs)
