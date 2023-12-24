"""
Implements a DQN learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.utils import ReplayBuffer, Logger, TestMetric, set_global_seed
from src.envs.utils import ExtraAction

import pandas as pd
import networkx as nx


class DQN:
    def __init__(
            self,
            envs,
            network,

            # Initial network parameters.
            init_network_params=None,
            init_weight_std=None,

            # DQN parameters
            double_dqn=True,
            update_target_frequency=10000,
            gamma=0.99,
            clip_Q_targets=False,

            # Replay buffer.
            replay_start_size=50000,
            replay_buffer_size=1000000,
            minibatch_size=32,
            update_frequency=1,

            # Learning rate
            update_learning_rate=True,
            initial_learning_rate=0,
            peak_learning_rate=1e-3,
            peak_learning_rate_step=10000,
            final_learning_rate=5e-5,
            final_learning_rate_step=200000,

            # Optional regularization.
            max_grad_norm=None,
            weight_decay=0,

            # Exploration
            update_exploration=True,
            initial_exploration_rate=1,
            final_exploration_rate=0.1,
            final_exploration_step=1000000,

            # Loss function
            adam_epsilon=1e-8,
            loss="mse",

            # Saving the agent
            save_network_frequency=10000,
            network_save_path='network',
            network_save_path_batch=None,

            # Testing the agent
            evaluate=True,
            test_envs=None,
            test_episodes=20,
            test_envs_batch=40,
            test_frequency=10000,
            test_grid_graph=False,
            test_save_path='test_scores',
            test_metric=TestMetric.ENERGY_ERROR,

            # Other
            randweight=False,
            logging=True,
            seed=None,
            verbose=True
    ):
        self.randweight = randweight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.double_dqn = double_dqn

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration,
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.random_low = 1
        self.random_high = 10
        self.random_weight = []
        self.verbose = verbose
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        if type(envs) != list:
            envs = [envs]
        self.envs = envs
        self.env, self.acting_in_reversible_spin_env = self.get_random_env()

        self.replay_buffers = {}
        for n_spins in set([env.action_space.n for env in self.envs]):
            self.replay_buffers[n_spins] = ReplayBuffer(self.replay_buffer_size)

        self.replay_buffer = self.get_replay_buffer_for_env(self.env)

        self.seed = random.randint(0, 1e6) if seed is None else seed

        for env in self.envs:
            set_global_seed(self.seed, env)

        self.network = network().to(self.device)
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params is not None:
            if self.verbose:
                print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std is not None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        if self.verbose:
                            print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)

                with torch.no_grad():
                    self.network.apply(init_weights)

        self.target_network = network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay)

        self.evaluate = evaluate
        if test_envs in [None, [None]]:
            # By default, test on the same environment(s) as are trained on.
            self.test_envs = self.envs
        else:
            if type(test_envs) != list:
                test_envs = [test_envs]
            self.test_envs = test_envs
        self.test_grid_graph = test_grid_graph
        self.test_envs_batch = test_envs_batch
        self.test_episodes = int(test_episodes)
        if self.test_grid_graph:
            self.test_episodes = 1
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        if not self.acting_in_reversible_spin_env:
            for env in self.envs:
                assert env.extra_action == ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."
            for env in self.test_envs:
                assert env.extra_action == ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."

        self.allowed_action_state = self.env.get_allowed_action_states()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path
        self.network_save_path_batch = network_save_path_batch

    def get_random_weight(self):
        self.random_weight = np.array(np.random.randint(self.random_low, self.random_high, size=self.env.n_spins))

    def get_random_env(self, envs=None):
        if envs is None:
            env = random.sample(self.envs, k=1)[0]
        else:
            env = random.sample(envs, k=1)[0]

        return env, env.reversible_spins

    def get_replay_buffer_for_env(self, env):
        return self.replay_buffers[env.action_space.n]

    def get_random_replay_buffer(self):
        return random.sample(self.replay_buffers.items(), k=1)[0][1]

    def learn(self, timesteps):
        test_scores_list = []
        if self.logging:
            logger = Logger()

        # Initialise the state
        state = torch.as_tensor(self.env.reset())
        if self.randweight:
            self.get_random_weight()
        score = 0
        losses_eps = []
        t1 = time.time()
        test_scores = []
        losses = []

        is_training_ready = False

        for timestep in range(timesteps):
            if not is_training_ready:
                if all([len(rb) >= self.replay_start_size for rb in self.replay_buffers.values()]):
                    if self.verbose:
                        print('\nAll buffers have {} transitions stored - training is starting!\n'.format(
                            self.replay_start_size))
                    is_training_ready = True
            # Choose action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate:
                self.update_lr(timestep)

            # Perform action in environment
            state_next, reward, done, _ = self.env.step(action, self.random_weight, adj_mask=True)

            score += reward

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            done = torch.as_tensor([done], dtype=torch.float)
            self.replay_buffer.add(state, action, reward, state_next, done)

            if done:
                # Reinitialise the state
                if self.verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep + 1),
                        self.env.current_step,
                        np.round(score, 3),
                        loss_str,
                        round(time.time() - t1, 3)))

                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                self.env, self.acting_in_reversible_spin_env = self.get_random_env()
                self.replay_buffer = self.get_replay_buffer_for_env(self.env)
                state = torch.as_tensor(self.env.reset())
                if self.randweight:
                    self.get_random_weight()
                score = 0
                losses_eps = []
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:
                # Update the main network
                if done:
                    # Sample a batch of transitions
                    transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep, loss])
                    losses_eps.append(loss)

                    if self.logging:
                        logger.add_scalar('Loss', loss, timestep)

                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep + 1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                test_score = self.test_agent(timestep)
                test_scores_list.append(test_score)
                if self.verbose:
                    print('\nTest score: {}\n'.format(np.round(test_score, 3)))

                if self.test_metric in [TestMetric.FINAL_CUT, TestMetric.MAX_CUT, TestMetric.CUMULATIVE_REWARD]:
                    best_network = all([test_score > score for t, score in test_scores])
                elif self.test_metric in [TestMetric.ENERGY_ERROR, TestMetric.BEST_ENERGY]:
                    best_network = all([test_score < score for t, score in test_scores])
                else:
                    raise NotImplementedError("{} is not a recognised TestMetric".format(self.test_metric))

                if best_network:
                    if self.network_save_path_batch is not None:
                        for path in self.network_save_path_batch:
                            path_main, path_ext = os.path.splitext(path)
                            path_main += "_best"
                            if path_ext == '':
                                path_ext += '.pth'
                            self.save(path_main + path_ext)

                test_scores.append([timestep + 1, test_score])

            if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
                path = self.network_save_path
                path_main, path_ext = os.path.splitext(path)
                path_main += str(timestep + 1)
                if path_ext == '':
                    path_ext += '.pth'
                self.save(path_main + path_ext)

        if self.logging:
            logger.save()

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'
        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print('test_scores saved to {}'.format(path))

        with open(self.losses_save_path, 'wb+') as output:
            pickle.dump(np.array(losses), output, pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print('losses saved to {}'.format(self.losses_save_path))
        return np.mean(test_scores_list)

    @torch.no_grad()
    def __only_bad_actions_allowed(self, state, network):
        x = (state[0, :] == self.allowed_action_state).nonzero()
        q_next = network(state.to(self.device).float())[x].max()
        return True if q_next < 0 else False

    def train_step(self, transitions):

        states, actions, rewards, states_next, dones = transitions
        if self.acting_in_reversible_spin_env:
            # Calculate target Q
            with torch.no_grad():
                if self.double_dqn:
                    greedy_actions = self.network(states_next.float()).argmax(1, True)
                    q_value_target = self.target_network(states_next.float()).gather(1, greedy_actions)
                else:
                    q_value_target = self.target_network(states_next.float()).max(1, True)[0]

        else:
            target_preds = self.target_network(states_next.float())
            disallowed_actions_mask = (states_next[:, 0, :] != self.allowed_action_state)

            # Calculate target Q, selecting ONLY ALLOWED ACTIONS greedily.
            with torch.no_grad():
                if self.double_dqn:
                    network_preds = self.network(states_next.float())
                    network_preds_allowed = network_preds.masked_fill(disallowed_actions_mask, -10000)
                    greedy_actions = network_preds_allowed.argmax(1, True)
                    q_value_target = target_preds.gather(1, greedy_actions)
                else:
                    q_value_target = target_preds.masked_fill(disallowed_actions_mask, -10000).max(1, True)[0]

        if self.clip_Q_targets:
            q_value_target[q_value_target < 0] = 0

        # Calculate TD target
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value
        q_value = self.network(states.float()).gather(1, actions)

        # Calculate loss
        loss = self.loss(q_value, td_target, reduction='mean')

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            action = self.predict(state)

        else:
            if self.acting_in_reversible_spin_env:
                # Random random spin.
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Flip random spin from that hasn't yet been flipped.
                x = (state[0, :] == self.allowed_action_state).nonzero()
                action = x[np.random.randint(0, len(x))].item()
        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
                timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
            )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (
                        self.final_learning_rate_step - self.peak_learning_rate_step)
            )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    @torch.no_grad()
    def predict(self, states, acting_in_reversible_spin_env=None):
        if acting_in_reversible_spin_env is None:
            acting_in_reversible_spin_env = self.acting_in_reversible_spin_env
        qs = self.network(states)
        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = qs.argmax().item()
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            if qs.dim() == 1:
                x = (states[0, :] == self.allowed_action_state).nonzero()
                actions = x[qs[x].argmax().item()].item()
            else:
                disallowed_actions_mask = (states[:, :, 0] != self.allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -10000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    def new_diretory(self, path):
        path = path.strip()
        path = path.rstrip("\\")
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return True
        else:
            return False

    # plot
    def save_test_graphs(self, obs_batch, timestep, reset=False, show_n=None, mask=None, verbose=False):
        Graphs = []
        plt.figure(figsize=(12, 8))
        n = obs_batch[0].shape[0]
        count = 0
        for obs in obs_batch:
            Graphs.append(nx.from_numpy_matrix(obs[1:n, :]))
            if reset:
                df = pd.DataFrame(data=obs)
                self.new_diretory('test_data//test_graph_data//time_' + str(timestep + 1))
                df.to_csv(
                    'test_data//test_graph_data//time_' + str(timestep + 1) + '//graph_' + str(count + 1) + '.csv')
            count += 1
        pos = [nx.spring_layout(G, iterations=100) for G in Graphs]
        if show_n is None:
            for i in range(4):
                plt.subplot(int('22' + str(i + 1)))
                nx.draw(Graphs[i], pos[i], node_size=100)
                if mask is not None:
                    nodelist = [mask[i][j] for j in np.where(mask[i, :] != -1)[-1].tolist()]
                    nx.draw_networkx_nodes(Graphs[i], pos[i], nodelist, node_size=100, node_color='r')
        else:
            if show_n % 2 == 0:
                row = int(show_n / 2)
                for i in range(show_n):
                    plt.subplot(int(str(row) + '2' + str(i + 1)))
                    nx.draw(Graphs[i], pos[i], node_size=100)
                    if mask is not None:
                        nodelist = [mask[i][j] for j in np.where(mask[i, :] != -1)[-1].tolist()]
                        nx.draw_networkx_nodes(Graphs[i], pos[i], nodelist=nodelist, node_size=100, node_color='r')
            else:
                raise Exception('show_n has to be a multiple of 2 since fig shows 2 graph per row.')
        if reset:
            plt.savefig('test_data//test_graph//graph_img_original_' + str(timestep + 1) + '.png', bbox_inches='tight')
        else:
            plt.savefig('test_data//test_graph//graph_img_' + str(timestep + 1) + '.png', bbox_inches='tight')
        if verbose:
            plt.show()

    def generate_test_envs(self, batch_size=None, grid_graph=False):
        i_test = 0
        if batch_size is None:
            batch_size = self.minibatch_size
        test_envs = np.array([None] * batch_size)
        obs_batch = []
        for i, env in enumerate(test_envs):
            if env is None and i_test < self.test_episodes:
                test_env, testing_in_reversible_spin_env = self.get_random_env(self.test_envs)
                obs = test_env.reset()
                test_env = deepcopy(test_env)
                test_envs[i] = test_env
                obs_batch.append(obs)
        if grid_graph:
            grid = nx.grid_graph((4, 5))
            for envs, obs in zip(test_envs, obs_batch):
                obs[1:obs.shape[1] + 1, :] = nx.to_numpy_array(grid)
                envs.matrix = nx.to_numpy_array(grid)
                envs.matrix_obs = nx.to_numpy_array(grid)
        return test_envs, obs_batch

    @torch.no_grad()
    def test_agent(self, timestep):
        batch_scores = np.zeros((self.test_episodes, self.test_envs_batch))
        test_scores, test_obs_batch = [], []
        action_batch = np.zeros((self.test_episodes, self.test_envs_batch, self.env.n_spins))
        action_batch[action_batch == 0] = -1
        i_comp = 0
        while i_comp < self.test_episodes:
            i = 0
            test_envs, obs_batch = self.generate_test_envs(batch_size=self.test_envs_batch,
                                                           grid_graph=self.test_grid_graph)
            test_obs_batch.append(obs_batch)
            for env, state in zip(test_envs, obs_batch):
                j = 0
                if env is not None:
                    done = False
                    while not done:
                        action = self.predict(torch.FloatTensor(state).to(self.device),
                                              acting_in_reversible_spin_env=False)
                        action_batch[i_comp][i][j] = action
                        state, rew, done, info = env.step(action, env.random_weight, adj_mask=True)
                        if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                            batch_scores[i_comp][i] += rew
                        j += 1
                else:
                    raise Exception('env gets None!')
                if self.test_metric == TestMetric.BEST_ENERGY:
                    batch_scores[i_comp][i] = env.best_energy
                elif self.test_metric == TestMetric.ENERGY_ERROR:
                    batch_scores[i_comp][i] = abs(env.best_energy - env.calculate_best()[0])
                elif self.test_metric == TestMetric.MAX_CUT:
                    # here
                    batch_scores[i_comp][i] = env.get_best_dsp()
                elif self.test_metric == TestMetric.FINAL_CUT:
                    batch_scores[i_comp][i] = env.calculate_dsp()
                i += 1
            test_scores.append(np.mean(batch_scores[i_comp, :]))
            i_comp += 1
        # best_index = random.choice(np.where(test_scores == np.max(test_scores))[0].tolist()) self.save_test_graphs(
        # test_obs_batch[best_index], timestep, mask=action_batch[best_index, :, :], verbose=verbose)
        return np.max(test_scores)

    def save(self, path='network.pth'):
        if os.path.splitext(path)[-1] == '':
            path + '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
