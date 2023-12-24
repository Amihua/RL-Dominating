import pickle
import numpy as np
import time
import pandas as pd


def greedy(G):
    N = len(G.nodes())
    Dom_set = []
    state = np.zeros(N)
    un_choose = np.where(state == 0)[0]
    while len(un_choose) > 0:
        degree = np.zeros(N)
        for i in G.degree(un_choose):
            v, deg = i
            degree[v] = deg
        remove_i = np.argmax(degree)
        state[remove_i] = 2
        for n in G.neighbors(remove_i):
            state[n] = 1
        Dom_set.append(remove_i)
        un_choose = np.where(state == 0)[0]
    res = len(np.where(state == 2)[0])
    return res


# load_loc = '../../data/ER_graphs/'
load_loc = '../../data/BA_graphs/'
# p = ['p0.1', 'p0.3', 'p0.5', 'p0.8']
p = ['m4', 'm8', 'm12', 'm18']
spins = [20, 40, 80, 100, 200, 300, 400, 500, 800]
res_list_p = []
time_list_p = []
for i in range(len(p)):
    # graph_path = [load_loc + str(spins[j]) + 'spins/ER_' + p[i] + '.pkl' for j in range(len(spins))]
    graph_path = [load_loc + str(spins[j]) + 'spins/BA_' + p[i] + '.pkl' for j in range(len(spins))]
    res_list_spins = []
    time_list_spins = []
    for path in graph_path:
        graphs = pickle.load(open(path, 'rb'))
        res_list_graphs = []
        time_list_graphs = []
        for g in graphs:
            t_start = time.time()
            res_list_graphs.append(greedy(g))
            time_list_graphs.append(time.time() - t_start)
        res_list_spins.append(np.mean(res_list_graphs))
        time_list_spins.append(np.mean(time_list_graphs))
    res_list_p.append(res_list_spins)
    time_list_p.append(time_list_spins)

pickle.dump(res_list_p, open('BA_res_list.pkl', "wb"))
pickle.dump(time_list_p, open('BA_time_list_p.pkl', "wb"))
res_df = pd.DataFrame(res_list_p, columns=['20', '40', '80', '100', '200', '300', '400', '500', '800'], index=p)
time_df = pd.DataFrame(time_list_p, columns=['20', '40', '80', '100', '200', '300', '400', '500', '800'], index=p)
res_df.to_pickle('BA_greedy_sol.pkl')
res_df.to_excel('BA_greedy_sol.xlsx')
time_df.to_pickle('BA_greedy_time.pkl')
time_df.to_excel('BA_greedy_time.xlsx')
