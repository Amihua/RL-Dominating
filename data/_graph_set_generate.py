import pickle
import random
import pandas as pd
import networkx as nx
import numpy as np


def openreadtxt(file_name):
    data = []
    file = open(file_name, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(' ')
        tmp_list[-1] = tmp_list[-1].replace('\n', '')
        data.append(tmp_list)
    return data


class SetGraphGrenerator:
    def __init__(self, n_spins, style, size, file_path, density=None, m=None, p=None, grid_n=None, grid_m=None):
        self.n_spins = n_spins
        self.density = density
        self.path = file_path
        self.set_size = size
        self.style = style
        self.m = m
        self.p = p
        self.grid_n = grid_n
        self.grid_m = grid_m

    def get_matrix(self):
        matrix = np.zeros((self.n_spins, self.n_spins))
        density = np.random.uniform()
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    w = random.choice([0, 1])
                    matrix[i, j] = w
                    matrix[j, i] = w
        return matrix

    def get_GRID_graph(self):
        if self.grid_n is not None and self.grid_m is not None:
            return nx.grid_graph((self.grid_n, self.grid_m))
        else:
            raise Exception('generate grid graph need parameter n and m!')

    def get_BA_graph(self):
        return nx.barabasi_albert_graph(self.n_spins, self.m)

    def get_ER_graph(self):
        g = nx.erdos_renyi_graph(self.n_spins, self.p)
        while not nx.is_connected(g) or len(g.edges) != 8944:
            g = nx.erdos_renyi_graph(self.n_spins, self.p)
        return g

    def get_TRIGRID_graph(self):
        return nx.triangular_lattice_graph(self.grid_n, self.grid_m)

    def get_HEXGRID_graph(self):
        return nx.hexagonal_lattice_graph(self.grid_n, self.grid_m)

    def get_graph_set(self):
        if self.style == 'random':
            return [self.get_matrix() for _ in range(self.set_size)]
        elif self.style == 'BA':
            return [self.get_BA_graph() for _ in range(self.set_size)]
        elif self.style == 'ER':
            return [self.get_ER_graph() for _ in range(self.set_size)]
        elif self.style == 'Grid':
            return [self.get_GRID_graph() for _ in range(self.set_size)]
        elif self.style == 'TriGrid':
            return [self.get_TRIGRID_graph() for _ in range(self.set_size)]
        elif self.style == 'HexGrid':
            return [self.get_HEXGRID_graph() for _ in range(self.set_size)]
        else:
            raise Exception('No style called ' + self.style)

    def generate_file(self):
        pickle.dump(self.get_graph_set(), open(self.path, "wb"))


if __name__ == '__main__':
    style = 'ER'
    # style = 'BA'
    n_spins = 400
    # density = 0.4
    # m = 18
    p = 0.1
    set_size = 1
    file_path = "ER_graphs/400spins_p0.1_opt8/ER_graphs.pkl"
    generator = SetGraphGrenerator(n_spins=n_spins, style=style, size=set_size, file_path=file_path, p=p)
    generator.generate_file()
    # n, m = 29, 29
    # for n_spins in [20, 40, 80, 100, 200, 300, 400, 500, 800]:
    #     for p in [0.1, 0.3, 0.5, 0.8]:
    #         file_path = "ER_graphs/" + str(n_spins) + 'spins/ER_p' + str(p) + '.pkl'
    #         # file_path = "../eval/checkpoints/20spins/20spins/BA_p0.8.pkl"
    #         generator = SetGraphGrenerator(n_spins=n_spins, style=style, size=set_size, file_path=file_path, p=p)
    #         # generator = SetGraphGrenerator(n_spins=n_spins, style=style, size=set_size, file_path=file_path, m=m)
    #         generator.generate_file()

    # for n_spins in [20, 40, 80, 100, 200, 300, 400, 500, 800]:
    #     for m in [4, 8, 12, 18]:
    #         file_path = "BA_graphs/" + str(n_spins) + 'spins/BA_m' + str(m) + '.pkl'
    #         generator = SetGraphGrenerator(n_spins=n_spins, style=style, size=set_size, file_path=file_path, m=m)
    #         generator.generate_file()


    # TriGrid: 20=4*6 40=7*8 60=9*10 78=12*10 102=16*10 200=18*19 305=20*27 406=27*27 496=30*30

    # HexGrid: 22=2*3 38=4*3 58=4*5 82=6*5 96=6*6 196=10*8 306=13*10 418=14*13 510=15*15

    # G = nx.erdos_renyi_graph(20, p=0.8)
    # print(len(G.nodes()))
    # nx.draw(G)
    # plt.show()
    # res = open('validation/BA_m12_ER_p60/20spins_BA_100graphs.pkl', 'rb')
    # pd = pickle.load(res)

    ######################################################
    #      This part used to generate real network .pkl  #
    ######################################################
    # data = openreadtxt('C:\\Users\\Administrator\\PycharmProjects\\DominatingSet\\real_dataset\\facebook_combined.txt')
    # # print(data)
    # G = nx.Graph()
    # for edge in data:
    #     G.add_edge(int(edge[0]), int(edge[1]))
    # print(G)
    # pickle.dump([G], open('egoFacebook.pkl', "wb"))
