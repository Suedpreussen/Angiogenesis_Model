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
class Model
    def __init__
        Set up lattice
        Compute source vector
        Compute other physical values
        Create pandas dataframes
    def update_pandas_data
    def update_networkx_data
    def check_Kirchhoffs_and_Murrays_law
    def compute_energy_dissipation
    def create_experiment_settings_log
    def dump_data_to_database
    def draw_graph
    def run_simulation
"""


class Model:
    """Main class"""
    def __init__(self, number_of_rows_or_columns: int, shape_of_boundaries="square", type_of_lattice="square"):
        """Class constructor"""
        if type_of_lattice == "square":
            assert number_of_rows_or_columns % 2 != 0, "Square model needs to have an odd number of rows or columns to get a central node."

        # save given model parameters
        self.__number_of_rows_or_columns = number_of_rows_or_columns
        self.__shape_of_boundaries = shape_of_boundaries
        self.__type_of_lattice = type_of_lattice


        """Set up lattice"""
        if type_of_lattice == "square":
            graph = nx.grid_graph(dim=(number_of_rows_or_columns, number_of_rows_or_columns))
        elif type_of_lattice == "triangular":
            graph = nx.triangular_lattice_graph(number_of_rows_or_columns, number_of_rows_or_columns, periodic=True)

        # compute incidence matrix of the graph
        inc_mtx = nx.incidence_matrix(graph)
        inc_mtx_dense = scipy.sparse.csr_matrix.todense(inc_mtx)
        inc_mtx_dense_int = inc_mtx_dense.astype(int)

        # change incidence matrix to a directed one
        for row in inc_mtx_dense_int.T:
            ones_count = 0
            row_element_count = 0
            for element in row:
                if element == 1:
                    ones_count += 1
                    if ones_count == 2:
                        row[row_element_count] *= -1
                row_element_count += 1

        # save relevant quantities as attributes
        self.graph = graph
        nodes_list = graph.nodes()
        edges_list = graph.edges()
        self.nodes_list = nodes_list
        self.edges_list = edges_list
        incidence_matrix = inc_mtx_dense_int
        self.incidence_matrix = incidence_matrix


        """Compute source vector"""
        # how to assess source_value?
        # should it be another argument in __init__ or just some derivative of number_of_rows_or_columns?
        # source in the center of the square lattice -- eye retina model
        # works only for odd number of rows/columns -- only then a central node exists
        nodes_dim = graph.number_of_nodes()
        source_value = (number_of_rows_or_columns-1)**2/ 2
        self.source_value = source_value
        if type_of_lattice == "square" and shape_of_boundaries == "square":
            source_list = np.zeros(nodes_dim)
            source_list[int((nodes_dim - 1) / 2)] = source_value  # source in the center
            number_of_boundary_nodes = 4 * np.sqrt(nodes_dim) - 4
            last_index = int(np.sqrt(nodes_dim) - 1)
            iterator = 0
            for node in graph.nodes:  # accessing nodes on the boundaries of the network
                if node[0] == 0 or node[0] == last_index or node[1] == 0 or node[1] == last_index:
                    source_list[iterator] = -source_value / number_of_boundary_nodes
                iterator += 1
        self.source_list = source_list


        """Compute other physical values"""
        # nodes-space vectors: source, pressure
        # edges-space vectors: conductivity, length, flow, pressure difference
        edges_dim = np.shape(incidence_matrix)[1]

        incidence_T = incidence_matrix.transpose()
        incidence_T_inv = np.linalg.pinv(incidence_T)
        incidence_inv = np.linalg.pinv(incidence_matrix)

        epsilon = 0.8
        radii_list = np.ones(edges_dim) + np.random.default_rng().uniform(-epsilon, epsilon,
                                                                          edges_dim)  # ones + stochastic noise
        conductivity_list = 0.3 * np.float_power(radii_list, 4)
        length_list = 0.8 * np.ones(edges_dim)

        # Q = (delta^T)^-1 * S
        flow_list = np.dot(source_list, incidence_T_inv)

        # delta_p = K/L * Q
        pressure_diff_list = flow_list * length_list * (1 / conductivity_list)
        pressure_list = np.dot(pressure_diff_list, incidence_inv)

        # x = delta^T * K/L *delta
        x = incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T
        x_dagger = np.linalg.pinv(x)  # Penrose pseudo-inverse

        # Q = K/L * delta * (delta^T * K/L * delta)^dagger * S
        flow_list = source_list @ x_dagger @ incidence_matrix @ np.diag(conductivity_list) @ np.diag(1 / length_list)

        # save to attributes
        self.conductivity_list = conductivity_list
        self.length_list = length_list
        self.flow_list = flow_list
        self.pressure_diff_list = pressure_diff_list
        self.pressure_list = pressure_list
        self.x = x
        self.incidence_T = incidence_T
        self.incidence_inv = incidence_inv


        "Create pandas dataframes"
        # creating data frames
        nodes_data = pd.DataFrame(nodes_list)
        edges_data = pd.DataFrame(edges_list)

        # filling up the data frames
        if np.shape(nodes_data)[1] == 1:  # if nodes are indexing by one int
            nodes_data.columns = ['nodes']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
        elif np.shape(nodes_data)[1] == 2:  # if nodes are indexing by two ints
            nodes_data.columns = ['no-', '-des']
            nodes_data['pressure'] = pressure_list
            nodes_data['source'] = source_list
        edges_data.columns = ['ed-', '-ges']
        edges_data['conductivity'] = conductivity_list
        edges_data['flow'] = np.abs(flow_list)
        edges_data['press_diff'] = pressure_diff_list

        # save to attributes
        self.nodes_data = nodes_data
        self.edges_data = edges_data

    def __update_pandas_data(self):
        self.edges_data['conductivity'] = self.conductivity_list
        self.edges_data['flow'] = np.abs(self.flow_list)
        self.edges_data['press_diff'] = self.pressure_diff_list
        self.nodes_data['pressure'] = self.pressure_list
        self.nodes_data['source'] = self.source_list

    def __update_networkx_data(self):
        # node_attrs = {tuple : dict, tuple: dict, ...} -- dict of (tuples as keys) and (dicts as values)
        node_attrs = dict(self.graph.nodes)
        iterator = 0
        for key in node_attrs:
            vals = {"pressure": self.pressure_list[iterator]}
            node_attrs[key] = vals
            iterator += 1
        nx.set_node_attributes(self.graph, node_attrs)
        # now for edges
        edge_attrs = dict(self.graph.edges)
        iterator = 0
        for key in edge_attrs:
            vals = {"conductivity": self.conductivity_list[iterator],
                    "flow": self.flow_list[iterator], "pressure_diff": self.pressure_diff_list[iterator]}
            edge_attrs[key] = vals
            iterator += 1
        nx.set_edge_attributes(self.graph, edge_attrs)

    def __check_kirchhoffs_and_murrays_law(self):
        index = 0
        successful_Kirchhoffs_nodes = 0
        successful_Murrays_nodes = 0
        alpha = 7 / 3
        for node in self.graph.nodes(data=False):
            flow_sum = 0
            radii_in_sum = 0
            radii_out_sum = 0
            for edge in self.graph.edges(node):  # implementing direction of flow to the undirected graph
                if np.sum(edge[0]) < np.sum(edge[1]):
                    flow_sum += self.graph[edge[0]][edge[1]]['flow']
                    radii_in_sum += np.float_power(self.graph[edge[0]][edge[1]]['conductivity'], -alpha / 4)
                else:
                    flow_sum -= self.graph[edge[0]][edge[1]]['flow']
                    radii_out_sum -= np.float_power(self.graph[edge[0]][edge[1]]['conductivity'], -alpha / 4)

            if -1e-11 < flow_sum - self.source_list[index] < 1e-11:  # checking for every node if the sum of inflows and ouflows yields zero
                successful_Kirchhoffs_nodes += 1
            else:
                pass
                print(flow_sum, '|', self.source_list[index], '|', print(node))
                # print("Kirchhoff's law at node {} NOT fulfilled!".format(node), flow_sum + source_list[index])

            if -1e-11 < np.abs(radii_in_sum - radii_out_sum) < 1e-11:  # checking M's law
                successful_Murrays_nodes += 1
            else:
                pass
                # print(np.abs(radii_in_sum - radii_out_sum) , '||', print(node))
            index += 1

        print("number of nodes fulfilling K's law:", successful_Kirchhoffs_nodes, 'out of', self.graph.number_of_nodes())
        print("number of nodes fulfilling Murray's law:", successful_Murrays_nodes, 'out of', self.graph.number_of_nodes())

        if successful_Kirchhoffs_nodes == self.graph.number_of_nodes():
            print("SUCCESS! Kirchhoff's law fulfilled!")

    def __compute_energy_dissipation(self, gamma, show_result=False):
        # calculating energy functional E = sum over edges L * Q^2 / K
        energy_list = self.length_list * self.flow_list * self.flow_list / self.conductivity_list
        energy = np.sum(energy_list)

        # checking cost constraint = sum over edges L * K^(1/gamma - 1)
        constraint = np.sum(self.length_list * np.float_power(self.conductivity_list, (1 / gamma - 1)))

        if show_result:
            print("Energy: ", energy)
            print("Constraint: ", constraint)

    def create_experiment_settings_log(self):
        pass

    def dump_data_to_database(self):
        pass

    def draw_graph(self, name: str, directory_name: str):
        number_of_rowscols = self.__number_of_rows_or_columns
        graph = self.graph
        conductivity_list = self.conductivity_list

        max = 26  # max value on the colour map bar
        cmap = plt.cm.magma_r

        # setting the layout for the graph visualisation
        if self.__type_of_lattice == "hexagonal":
            pos = nx.get_node_attributes(graph, 'pos')  # hexagonal rigid layout
        elif self.__type_of_lattice == "triangular":
            pos = nx.get_node_attributes(graph, 'pos')  # triangular rigid layout
        else:
            pos = dict(
                (number_of_rowscols, number_of_rowscols) for number_of_rowscols in graph.nodes())  # square rigid layout

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
            nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1 / 4) * 2, edge_color=conductivity_list, edge_cmap=cmap, edge_vmin=0, edge_vmax=max)
        elif 399 < len(conductivity_list):
            # nx.draw_networkx_nodes(graph, nodelist=(n-1, n-1), pos=pos, node_size=100 / (2 * n), node_color='black')
            nc = nx.draw_networkx_edges(graph, pos=pos, width=np.float_power(conductivity_list, 1 / 4) * 1.5, edge_color=conductivity_list, edge_cmap=cmap, edge_vmin=0, edge_vmax=max)
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

    def run_simulation(self, directory_name: str, a, b, gamma, delta, nu, c, r, dt, N, is_scaled=False, create_log=False, dump_data=False, create_graphs=False):
        """Main part"""

        # retrieving data from the object
        graph = self.graph
        source_list = self.source_list
        source_value = self.source_value
        pressure_list = self.pressure_list
        length_list = self.length_list
        conductivity_list = self.conductivity_list
        flow_list = self.flow_list
        pressure_diff_list = self.pressure_diff_list
        nodes_data = self.nodes_data
        edges_data = self.edges_data
        number_of_rowscols = self.__number_of_rows_or_columns
        incidence_matrix = self.incidence_matrix
        incidence_T = self.incidence_T
        incidence_inv = self.incidence_inv

        flow_hat = np.average(np.abs(flow_list))
        t = 0

        self.__update_pandas_data()
        self.__update_networkx_data()
        self.__check_kirchhoffs_and_murrays_law()

        # two control parameters
        rho = b / (r * gamma * delta)  # the ratio between the time scales for adaptation and growth
        print("RHO: ", rho)

        source_hat = source_value
        kappa = (c / a) * np.float_power((flow_hat / source_hat), 2 * gamma)  # a/c is the ratio between background growth rate and adaptation strength and the hatted quantities are typical scales for flow and source strength.
        print("KAPPA: ", kappa)

        # implementing scaling factors
        if is_scaled:
            source_list = source_list * np.exp(r * t * delta / 2)
            pressure_list = pressure_list * np.exp(r * t * nu / 2)
            length_list = length_list * np.exp(r * t / 2)
            conductivity_list = conductivity_list * np.exp(r * t * delta * gamma)
            flow_list = flow_list * np.exp(r * t * delta / 2)
            b = b + r * gamma * delta
            c = c * np.exp(-r * t * gamma * delta)
            print("time unit: ", 1 / b)

        #lagrange_multiplier = 0.01
        #flow_from_lagrange_optimisation = np.sqrt(lagrange_multiplier) * np.sqrt(1 / gamma + 1) * np.float_power(conductivity_list, 1 / (2 * gamma))

        # snapshot before the sim
        # draw graphs
        self.__update_pandas_data()
        self.draw_graph(f"graph_at_0_{N}", "graphs")


        # print log
        print(f"______n = 0________")
        print("Q_av: ", np.average(np.abs(flow_list)))
        self.__compute_energy_dissipation(gamma, show_result=True)
        print("Sum of conductivity: ", np.sum(conductivity_list))

        # list_of_dfs = []                     # container to store dfs at snapshots
        # list_of_dfs.append(edges_data)

        # MAIN LOOP
        for n in range(1, N + 1):
            t += dt

            # dK/dt = a*(q / q_hat)^(2*gamma) - b * K + c
            dK = dt * (np.float_power(a * (np.abs(flow_list) / flow_hat),
                                      (2 * gamma)) - b * conductivity_list + c * np.ones(len(flow_list)))
            # dK = dt * (np.float_power(a * (np.abs(flow_from_lagrange_optimisation) / flow_hat), (2 * gamma)) - b * conductivity_list + c * np.ones(len(flow_list)))
            conductivity_list += dK

            x = incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T
            x_dagger = np.linalg.pinv(incidence_matrix @ np.diag(1 / length_list) @ np.diag(conductivity_list) @ incidence_T)

            # q = K/L * delta * (delta^T * K/L * delta)^dagger * S
            flow_list = source_list @ x_dagger @ incidence_matrix @ np.diag(conductivity_list) @ np.diag(1 / length_list)
            pressure_diff_list = length_list * (1 / conductivity_list) * flow_list
            pressure_list = np.dot(pressure_diff_list, incidence_inv)
            self.__compute_energy_dissipation(gamma)

            # updating data in graph dicts
            self.__update_networkx_data()

            # sim snapshots
            if n == N or n == N / 16 or n == (2 * N) / 16 or n == (3 * N) / 16 or n == N / 4 or n == N / 2 or n == (3 * N) / 4 or n == N / 32 or n == (2 * N) / 32 or n == (3 * N) / N:
                # draw graphs
                self.__update_pandas_data()
                self.draw_graph(f"graph_at_{n}_{N}", "graphs")
                # print log
                print(f"________n = {n}________")
                print("Q_av: ", np.average(np.abs(flow_list)))
                self.__compute_energy_dissipation(gamma, show_result=True)
                print("Sum of conductivity: ", np.sum(conductivity_list))

            # dEdF_lam_dgdF = np.sum(flow_list) / np.sum(conductivity_list)    # checking first eq from lagrange optimisation
            # print(dEdF_lam_dgdF)

        print('simulation time: ', round(t * b, 3), "1/(b')  =  ", round(t, 3), "seconds")
        self.__update_pandas_data()
        self.__check_kirchhoffs_and_murrays_law()
