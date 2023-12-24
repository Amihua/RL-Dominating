import argparse
import os
import pickle

from train import run as train_run
from test import run as test_run
import optuna


class DSPModel:
    def __init__(self, n_spins, graph_type, args: argparse.Namespace, save_loc='../checkpoints/', load_loc='../data/'):
        self.n_spins = n_spins
        self.type = graph_type
        self.original_save_loc = save_loc
        self.save_loc = save_loc
        self.original_load_loc = load_loc
        self.args = args
        if n_spins <= 0:
            raise Exception('Parameter n_spins must larger than 0!')
        if graph_type not in ['ER', 'ER_SOL', 'BA', 'grid', 'tri', 'hex', 'random', 'large', 'whole']:
            raise Exception(
                "No type called {}, graph type must be 'ER', 'ER_SOL', 'BA', 'GRID', 'TRI', 'HEX', 'large', 'whole".format(graph_type))
        if type(args) != argparse.Namespace:
            raise TypeError("Parameter 'args' must be type 'argparse.Namespace'!")
        self.set_paths()

    def set_paths(self):
        self.save_loc = self.original_save_loc + self.type + '_graphs/' + str(self.n_spins) + 'spins/'
        self.load_loc = self.original_load_loc + self.type + '_graphs/'

    def train(self, timesteps, verbose=True):
        score = train_run(timesteps=timesteps, n_spins=self.n_spins, test_loc=self.load_loc, save_loc=self.save_loc,
                           args=self.args, verbose=verbose)
        return score

    def test(self, test_model=None):
        if test_model not in [None, '20', '100']:
            assert "No model name called" + test_model + " ! Please set test_model in ['None', '20', '100']"
        if self.type == 'ER':
            p = ['p0.1', 'p0.3', 'p0.5', 'p0.8']
            spins = [20, 40, 80, 100, 200, 300, 400, 500, 800]
            for i in range(len(p)):
                graph_path = [self.load_loc + str(spins[j]) + 'spins/ER_' + p[i] + '.pkl' for j in range(len(spins))]
                for spin, path in zip(spins, graph_path):
                    max_batch_size = None
                    if spin >= 400:
                        max_batch_size = 5
                    save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                    test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                             args=self.args, p=p[i], model_name=test_model)
        elif self.type == 'BA':
            m = ['m4', 'm8', 'm12', 'm18']
            spins = [20, 40, 80, 100, 200, 300, 400, 500, 800]
            for i in range(len(m)):
                graph_path = [self.load_loc + str(spins[j]) + 'spins/BA_' + m[i] + '.pkl' for j in range(len(spins))]
                for spin, path in zip(spins, graph_path):
                    max_batch_size = None
                    if spin >= 400:
                        max_batch_size = 5
                    save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                    test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                             args=self.args, m=m[i], model_name=test_model)

        elif self.type == 'random':
            spins = [20, 40, 80, 100, 200, 300, 400, 500, 800]
            graph_path = [self.load_loc + str(spins[j]) + 'spins/random_graphs.pkl' for j in range(len(spins))]
            for spin, path in zip(spins, graph_path):
                max_batch_size = None
                if self.n_spins >= 400:
                    max_batch_size = 5
                save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                         args=self.args, model_name=test_model)

        elif self.type == 'ER_SOL':
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/ER_' + file + '.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args, model_name=test_model)

        elif self.type == 'large':
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/graph.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args,
                         model_name=test_model)

        elif self.type in ['grid', 'tri', 'hex']:
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/graph.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args,
                         model_name=test_model)

    def BA_ER_train_test(self, train_steps):
        if self.type == 'whole':
            graph_type = ['BA', 'ER']
            for ttype in graph_type:
                self.type = ttype
                self.set_paths()
                self.train(timesteps=train_steps)
                self.test()
        else:
            raise TypeError(
                "function BA_ER_train_test must used for graph type 'whole' , but got type '" + self.type + "'!")

    def para_optim_train(self, trail):
        gamma = trail.suggest_float('gamma', 0.95, 1)
        init_weight_std = trail.suggest_float('init_weight_std', 1e-3, 1)
        replay_buffer_size = trail.suggest_int('replay_buffer_size', 1000, 10000)
        update_target_frequency = trail.suggest_int('update_target_frequency', 500, 5000)
        lr = trail.suggest_float('lr', 1e-6, 1)
        minibatch_size = trail.suggest_int('minibatch_size', 16, 64)
        final_exploration_rate = trail.suggest_float('final_exploration_rate', 0, 0.2)
        parser.set_defaults(gamma=gamma)
        parser.set_defaults(init_weight_std=init_weight_std)
        parser.set_defaults(replay_buffer_size=replay_buffer_size)
        parser.set_defaults(update_target_frequency=update_target_frequency)
        parser.set_defaults(lr=lr)
        parser.set_defaults(minibatch_size=minibatch_size)
        parser.set_defaults(final_exploration_rate=final_exploration_rate)
        self.args = parser.parse_args()
        scores = self.train(timesteps=50000, verbose=False)
        return scores

    def optim(self):
        storage_name = 'sqlite:///optuna.db'
        study = optuna.create_study(direction="maximize",
                                    study_name="DSP_OPTIM", storage=storage_name, load_if_exists=True
                                    )
        study.optimize(self.para_optim_train, n_trials=200)
        best_para = study.best_params
        fw = open(self.save_loc + 'best_model_para.pkl', 'wb')
        pickle.dump(best_para, fw)
        fw.close()
        parser.set_defaults(gamma=best_para['gamma'])
        parser.set_defaults(init_weight_std=best_para['init_weight_std'])
        parser.set_defaults(replay_buffer_size=best_para['replay_buffer_size'])
        parser.set_defaults(update_target_frequency=best_para['update_target_frequency'])
        parser.set_defaults(lr=['lr'])
        parser.set_defaults(minibatch_size=best_para['minibatch_size'])
        parser.set_defaults(final_exploration_rate=best_para['final_exploration_rate'])
        self.args = parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSP_args')
    parser.add_argument('--gamma', type=float, default=0.98, help='Decrease rate of reward.')
    parser.add_argument('--norm_reward', default=False, help='Reward of reinforcement learning normalization or not.')
    parser.add_argument('--stag_punishment', default=1,
                        help='Punishment of steps of reinforcement learning or not.')
    parser.add_argument('--if_weight', default=False, help='Weighted graph experiments.')
    parser.add_argument('--init_weight_std', type=float, default=0.01, help='Init weight std of linear.')
    parser.add_argument('--replay_buffer_size', type=int, default=10000, help='Size of replay buffer.')
    parser.add_argument('--update_target_frequency', type=int, default=2000, help='Frequency of update target network.')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr.')
    parser.add_argument('--minibatch_size', type=int, default=32, help='Mini batch size')
    parser.add_argument('--initial_exploration_rate', type=float, default=1, help='Initial exploration rate.')
    parser.add_argument('--final_exploration_rate', type=float, default=0.05, help='Final exploration rate.')
    parser.add_argument('--final_exploration_step', type=float, default=800000, help='Final exploration step.')
    parser.add_argument('--grid', default=False, help='Grid tri or hex grid graphs experiments.')
    args = parser.parse_args()
    model = DSPModel(n_spins=100, graph_type='BA', args=args)
    # model.optim()
    model.train(timesteps=1600000)
    # for ttpye in ['ER_SOL']:
    #     model.type = ttpye
    #     model.set_paths()
    #     model.test(test_model='20')
