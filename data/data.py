import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd

# file_name = os.listdir('ER_SOL_graphs')
# for file in file_name:
#     G_set = []
#     v_num = int(file[file.index('V')+1:file.index('E')])
#     for index in range(10):
#         df = pd.read_csv('ER_SOL_graphs/' + file + '/Problem' + str(index) + '.dat', header=None)
#         data = df.iloc[df.shape[0]-v_num:df.shape[0], :]
#         adj_list = [[] for i in range(v_num)]
#         for i in range(v_num):
#             for string in data.iloc[i, :][0]:
#                 if string != ' ':
#                     adj_list[i].append(int(string))
#         adj = np.array(adj_list)
#         G_set.append(nx.from_numpy_matrix(adj))
#     pickle.dump(G_set, open('ER_SOL_graphs/' + file + '/ER_' + file + '.pkl', "wb"))

# g_label = ['anna', 'david', 'homer', 'huck']
# for name in g_label:
#     G = nx.read_edgelist('large_graphs/' + name + '.txt', create_using=nx.Graph())
#     pickle.dump([G], open('large_graphs/' + name + '/graph.pkl', "wb"))

# G = nx.karate_club_graph()
# pickle.dump([G], open('large_graphs/karate/graph.pkl', "wb"))
