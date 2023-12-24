import os
import pickle
import networkx as nx
import time
import numpy as np
import scipy as sp
import pandas as pd
import torch

from collections import namedtuple
from copy import deepcopy

import src.envs.core as ising_env
from src.envs.utils import (SingleGraphGenerator, SpinBasis)
from src.agents.solver import Network, Greedy


####################################################
# TESTING ON GRAPHS
####################################################

def test_network(network, env_args, graphs_test, device=None, step_factor=1, batched=True,
                 n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if batched:
        return __test_network_batched(network, env_args, graphs_test, device, step_factor,
                                      n_attempts, return_raw, return_history, max_batch_size)
    else:
        if max_batch_size is not None:
            print("Warning: max_batch_size argument will be ignored for when batched=False.")
        return __test_network_sequential(network, env_args, graphs_test, step_factor,
                                         n_attempts, return_raw, return_history)


def __test_network_batched(network, env_args, graphs_test, device=None, step_factor=1,
                           n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)

    # HELPER FUNCTION FOR NETWORK TESTING

    acting_in_reversible_spin_env = env_args['reversible_spins']

    if env_args['reversible_spins']:
        # If MDP is reversible, both actions are allowed.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = (0, 1)
        elif env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = (1, -1)
    else:
        # If MDP is irreversible, only return the state of spins that haven't been flipped.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = 0
        if env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = 1

    def predict(states):

        qs = network(states)

        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = [qs.argmax().item()]
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            if qs.dim() == 1:
                x = (states.squeeze()[:, 0] == allowed_action_state).nonzero()
                actions = [x[qs[x].argmax().item()].item()]
            else:
                disallowed_actions_mask = (states[:, :, 0] != allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -1000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    # NETWORK TESTING

    results = []
    results_raw = []
    if return_history:
        history = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1
    all_greedy_action_list = []
    results_master = []
    # model_res = []
    # greedy_res = []
    for j, test_graph in enumerate(graphs_test):

        i_comp = 0
        i_batch = 0
        t_total = 0

        n_spins = test_graph.shape[0]
        n_steps = int(n_spins * step_factor)

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)

        print("Running greedy solver with +1 initialisation of spins...", end="...")
        # Calculate the greedy cut with all spins initialised to +1
        t_greedy_start = time.time()
        greedy_env = deepcopy(test_env)
        greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))
        # print('----------------------')
        # print(greedy_env.random_weight)
        test_test_env = deepcopy(greedy_env)
        greedy_agent = Greedy(greedy_env)
        _, greedy_action_list = greedy_agent.solve(random_weight=greedy_env.random_weight)
        # print(greedy_action_list)
        greedy_res = (len(greedy_action_list))

        greedy_single_cut = greedy_env.get_best_cut()
        greedy_single_spins = greedy_env.best_spins
        t_greedy = time.time() - t_greedy_start
        print("done.")

        if return_history:
            actions_history = []
            rewards_history = []
            scores_history = []

        best_cuts = []
        init_spins = []
        best_spins = []

        greedy_cuts = []
        greedy_spins = []

        while i_comp < n_attempts:
            if max_batch_size is None:
                batch_size = 1
            else:
                batch_size = min(n_attempts - i_comp, max_batch_size)

            i_comp_batch = 0

            if return_history:
                actions_history_batch = [[None] * batch_size]
                rewards_history_batch = [[None] * batch_size]
                scores_history_batch = []

            test_envs = [None] * batch_size
            best_cuts_batch = [-1e3] * batch_size
            init_spins_batch = [[] for _ in range(batch_size)]
            best_spins_batch = [[] for _ in range(batch_size)]

            greedy_envs = [None] * batch_size
            greedy_cuts_batch = []
            greedy_spins_batch = []

            obs_batch = [None] * batch_size

            print("Preparing batch of {} environments for graph {}.".format(batch_size, j), end="...")

            for i in range(batch_size):
                env = deepcopy(test_test_env)
                obs_batch[i] = env.reset(reset_weight=False)
                # print(env.random_weight)
                test_envs[i] = env
                greedy_envs[i] = deepcopy(env)
                init_spins_batch[i] = env.best_spins
            if return_history:
                scores_history_batch.append([env.calculate_score() for env in test_envs])

            print("done.")

            # Calculate the max cut acting w.r.t. the network
            t_start = time.time()

            # pool = mp.Pool(processes=16)

            k = 0
            model_res = 0
            while i_comp_batch < batch_size:
                t1 = time.time()
                # Note: Do not convert list of np.arrays to FloatTensor, it is very slow!
                # see: https://github.com/pytorch/pytorch/issues/13918
                # Hence, here we convert a list of np arrays to a np array.
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                actions = predict(obs_batch)
                model_res += 1
                obs_batch = []

                if return_history:
                    scores = []
                    rewards = []

                i = 0
                for env, action in zip(test_envs, actions):
                    if env is not None:
                        obs, rew, done, info = env.step(action, env.random_weight, adj_mask=True)

                        if return_history:
                            scores.append(env.calculate_score())
                            rewards.append(rew)

                        if not done:
                            obs_batch.append(obs)
                        else:
                            best_cuts_batch[i] = env.get_best_cut()
                            best_spins_batch[i] = env.best_spins
                            i_comp_batch += 1
                            i_comp += 1
                            test_envs[i] = None
                    i += 1
                    k += 1

                if return_history:
                    actions_history_batch.append(actions)
                    scores_history_batch.append(scores)
                    rewards_history_batch.append(rewards)
            # model_res.append(temp)

                # print("\t",
                #       "Par. steps :", k,
                #       "Env steps : {}/{}".format(k/batch_size,n_steps),
                #       'Time: {0:.3g}s'.format(time.time()-t1))

            t_total += (time.time() - t_start)
            i_batch += 1
            print("Finished agent testing batch {}.".format(i_batch))

            if env_args["reversible_spins"]:
                print("Running greedy solver with {} random initialisations of spins for batch {}...".format(batch_size,
                                                                                                             i_batch),
                      end="...")

                for env in greedy_envs:
                    Greedy(env).solve()
                    cut = env.get_best_cut()
                    greedy_cuts_batch.append(cut)
                    greedy_spins_batch.append(env.best_spins)

                print("done.")

            if return_history:
                actions_history += actions_history_batch
                rewards_history += rewards_history_batch
                scores_history += scores_history_batch

            best_cuts += best_cuts_batch
            init_spins += init_spins_batch
            best_spins += best_spins_batch

            if env_args["reversible_spins"]:
                greedy_cuts += greedy_cuts_batch
                greedy_spins += greedy_spins_batch

            # print("\tGraph {}, par. steps: {}, comp: {}/{}".format(j, k, i_comp, batch_size),
            #       end="\r" if n_spins<100 else "")

        i_best = np.argmax(best_cuts)
        best_cut = best_cuts[i_best]
        sol = best_spins[i_best]

        mean_cut = np.mean(best_cuts)

        if env_args["reversible_spins"]:
            idx_best_greedy = np.argmax(greedy_cuts)
            greedy_random_cut = greedy_cuts[idx_best_greedy]
            greedy_random_spins = greedy_spins[idx_best_greedy]
            greedy_random_mean_cut = np.mean(greedy_cuts)
        else:
            greedy_random_cut = greedy_single_cut
            greedy_random_spins = greedy_single_spins
            greedy_random_mean_cut = greedy_single_cut

        print(
            'Graph {}, SOL: DSP: {}, greedy: {}.\t ||TIME: DSP_time:{}s '
            'greedy_time:{}\t\t\t'.format(
                j, model_res, greedy_res, np.round(t_total, 4),
                np.round(t_greedy, 4)))

        best_n = n_spins + 1
        for answer in best_spins:
            if len(np.where(answer == -1)[0].tolist()) < best_n:
                best_n = len(np.where(answer == -1)[0].tolist())
        best_n_greedy = len(np.where(greedy_random_spins == -1)[0].tolist())

        results_master.append([best_cut, greedy_random_cut,
                               np.round(t_total, 4), np.round(t_greedy, 4),
                               model_res, greedy_res
                               ])

        results.append([best_cut, sol,
                        mean_cut,
                        greedy_single_cut, greedy_single_spins,
                        greedy_random_cut, greedy_random_spins,
                        greedy_random_mean_cut,
                        np.round(t_total, 4), np.round(t_greedy, 4)])

        results_raw.append([init_spins,
                            best_cuts, best_spins,
                            greedy_cuts, greedy_spins])

        if return_history:
            history.append([np.array(actions_history).T.tolist(),
                            np.array(scores_history).T.tolist(),
                            np.array(rewards_history).T.tolist(), greedy_action_list])

    results = pd.DataFrame(data=results, columns=["cut", "sol",
                                                  "mean cut",
                                                  "greedy (+1 init) cut", "greedy (+1 init) sol",
                                                  "greedy (rand init) cut", "greedy (rand init) sol",
                                                  "greedy (rand init) mean cut",
                                                  "time", "greedy_time"])

    results_raw = pd.DataFrame(data=results_raw, columns=["init spins",
                                                          "cuts", "sols",
                                                          "greedy cuts", "greedy sols"])

    results_master = pd.DataFrame(data=results_master,
                                  columns=['DSP_scores', 'greedy_scores', 'DSP_time', 'greedy_time', 'DSP_sol',
                                           'greedy_sol'])

    if return_history:
        history = pd.DataFrame(data=history, columns=["actions", "scores", "rewards", "greedy_actions"])

    if return_raw == False and return_history == False:
        return results
    else:

        # ret = [results]
        # if return_raw:
        #     ret.append(results_raw)
        # if return_history:
        #     ret.append(history)
        return results_master


def __test_network_sequential(network, env_args, graphs_test, step_factor=1,
                              n_attempts=50, return_raw=False, return_history=False):
    if return_raw or return_history:
        raise NotImplementedError("I've not got to this yet!  Used the batched test script (it's faster anyway).")

    results = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for i, test_graph in enumerate(graphs_test):

        n_steps = int(test_graph.shape[0] * step_factor)

        best_cut = -1e3
        best_spins = []

        greedy_random_cut = -1e3
        greedy_random_spins = []

        greedy_single_cut = -1e3
        greedy_single_spins = []

        times = []

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        net_agent = Network(network, test_env,
                            record_cut=False, record_rewards=False, record_qs=False)

        greedy_env = deepcopy(test_env)
        greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))
        greedy_agent = Greedy(greedy_env)

        greedy_agent.solve(random_weight=greedy_env.random_weight)

        greedy_single_cut = greedy_env.get_best_cut()
        greedy_single_spins = greedy_env.best_spins

        for k in range(n_attempts):

            net_agent.reset(clear_history=True, reset_weight=False)
            greedy_env = deepcopy(test_env)
            greedy_agent = Greedy(greedy_env)

            tstart = time.time()
            net_agent.solve(random_weight=greedy_env.random_weight)
            times.append(time.time() - tstart)

            cut = test_env.get_best_cut()
            if cut > best_cut:
                best_cut = cut
                best_spins = test_env.best_spins

            greedy_agent.solve()

            greedy_cut = greedy_env.get_best_cut()
            if greedy_cut > greedy_random_cut:
                greedy_random_cut = greedy_cut
                greedy_random_spins = greedy_env.best_spins

            # print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
            #     i + 1, k, n_attemps, best_cut, greedy_random_cut, greedy_single_cut),
            #     end="\r")
            print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
                i + 1, k, n_attempts, best_cut, greedy_random_cut, greedy_single_cut),
                end=".")

        results.append([best_cut, best_spins,
                        greedy_single_cut, greedy_single_spins,
                        greedy_random_cut, greedy_random_spins,
                        np.mean(times)])

    return pd.DataFrame(data=results, columns=["cut", "sol",
                                               "greedy (+1 init) cut", "greedy (+1 init) sol",
                                               "greedy (rand init) cut", "greedy (rand init) sol",
                                               "time"])


####################################################
# LOADING GRAPHS
####################################################

Graph = namedtuple('Graph', 'name n_vertices n_edges matrix bk_val bk_sol')


def load_graph(graph_dir, graph_name):
    inst_loc = os.path.join(graph_dir, 'instances', graph_name + '.mc')
    val_loc = os.path.join(graph_dir, 'bkvl', graph_name + '.bkvl')
    sol_loc = os.path.join(graph_dir, 'bksol', graph_name + '.bksol')

    vertices, edges, matrix = 0, 0, None
    bk_val, bk_sol = None, None

    with open(inst_loc) as f:
        for line in f:
            arr = list(map(int, line.strip().split(' ')))
            if len(arr) == 2:  # contains the number of vertices and edges
                n_vertices, n_edges = arr
                matrix = np.zeros((n_vertices, n_vertices))
            else:
                assert type(matrix) == np.ndarray, 'First line in file should define graph dimensions.'
                i, j, w = arr[0] - 1, arr[1] - 1, arr[2]
                matrix[[i, j], [j, i]] = w

    with open(val_loc) as f:
        bk_val = float(f.readline())

    with open(sol_loc) as f:
        bk_sol_str = f.readline().strip()
        bk_sol = np.array([int(x) for x in list(bk_sol_str)] + [np.random.choice([0, 1])])  # final spin is 'no-action'

    return Graph(graph_name, n_vertices, n_edges, matrix, bk_val, bk_sol)


def load_graph_set(graph_save_loc):
    graphs_test = pickle.load(open(graph_save_loc, 'rb'))

    def graph_to_array(g):
        if type(g) == nx.Graph:
            g = nx.to_numpy_array(g)
        elif type(g) == sp.sparse.csr_matrix:
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_loc))
    return graphs_test


####################################################
# FILE UTILS
####################################################

def mk_dir(export_dir, quite=False):
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
            print('created dir: ', export_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        print('dir already exists: ', export_dir)
