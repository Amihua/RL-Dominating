import argparse
import os

import matplotlib.pyplot as plt
import torch

import src.envs.core as ising_env
from experiments.utils import test_network, load_graph_set
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            Observable)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns

    plt.style.use('seaborn')
except ImportError:
    pass


def run(save_loc, graph_save_loc, args: argparse.Namespace, batched=True,
        max_batch_size=None, p=None, m=None, model_name=None):
    data_folder = os.path.join(save_loc, 'data')
    if model_name is not None:
        network_save_path = save_loc + 'network_best_' + model_name + '.pth'
    else:
        network_save_path = save_loc + 'network_best_20.pth'

    # print("network params :", network_save_path)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

    step_factor = 1
    env_args = {'observables': [Observable.SPIN_STATE],
                'reward_signal': RewardSignal.DENSE,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.DSP,
                'spin_basis': SpinBasis.BINARY,
                'norm_rewards': args.norm_reward,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': args.stag_punishment,
                'basin_reward': None,
                'reversible_spins': False,
                'ifweight': args.if_weight}

    graphs_test = load_graph_set(graph_save_loc)

    test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(graphs_test[0]),
                              graphs_test[0].shape[0] * step_factor,
                              **env_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    # print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_path, map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    # print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))
    # results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
    #                                              return_raw=True, return_history=True,
    #                                              batched=batched, max_batch_size=max_batch_size)
    results = test_network(network, env_args, graphs_test, device, step_factor,
                           return_raw=True, return_history=True,
                           batched=batched, max_batch_size=max_batch_size)
    if p is not None:
        results.to_pickle(save_loc + p + 'test_res.pkl')
        results.to_excel(save_loc + p + 'test_res.xlsx')
    elif m is not None:
        results.to_pickle(save_loc + m + 'test_res.pkl')
        results.to_excel(save_loc + m + 'test_res.xlsx')
    else:
        results.to_pickle(save_loc + 'test_res.pkl')
        results.to_excel(save_loc + 'test_res.xlsx')

    # results_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    # results_raw_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    # history_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"

    # for res, fname, label in zip([results, results_raw, history],
    #                              [results_fname, results_raw_fname, history_fname],
    #                              ["results", "results_raw", "history"]):
    #     save_path = os.path.join(data_folder, fname)
    #     res.to_pickle(save_path)
        # print("{} saved to {}".format(label, save_path))


if __name__ == "__main__":
    run()
