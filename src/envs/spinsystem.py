import random
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy, copy
from operator import matmul

import numpy as np
import torch.multiprocessing as mp
from numba import jit, float64, int64, int32

from src.envs.utils import (EdgeType,
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            SpinBasis,
                            DEFAULT_OBSERVABLES,
                            GraphGenerator,
                            RandomGraphGenerator,
                            HistoryBuffer)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class SpinSystemFactory(object):

    @staticmethod
    def get(graph_generator=None,
            max_steps=20,
            observables=DEFAULT_OBSERVABLES,
            reward_signal=RewardSignal.DENSE,
            extra_action=ExtraAction.PASS,
            optimisation_target=OptimisationTarget.ENERGY,
            spin_basis=SpinBasis.SIGNED,
            norm_rewards=False,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None,  # None means an infinite horizon.
            stag_punishment=None,  # None means no punishment for re-visiting states.
            basin_reward=None,  # None means no reward for reaching a local minima.
            reversible_spins=True,  # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            init_snap=None,
            seed=None,
            ifweight=False):

        if graph_generator.biased:
            return SpinSystemBiased(graph_generator, max_steps,
                                    observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                    norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                    reversible_spins,
                                    init_snap, seed, ifweight)
        else:
            return SpinSystemUnbiased(graph_generator, max_steps,
                                      observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                      norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                      reversible_spins,
                                      init_snap, seed, ifweight)


class SpinSystemBase(ABC):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            return np.random.choice(self.actions, n)

    class observation_space():
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal=RewardSignal.DENSE,
                 extra_action=ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.DSP,
                 spin_basis=SpinBasis.SIGNED,
                 norm_rewards=False,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_spins=False,
                 init_snap=None,
                 seed=None,
                 ifweight=False):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping spins.
            init_snap: Optional snapshot to load spin system into pre-configured state for MCTS.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."

        self.observables = list(enumerate(observables))

        self.extra_action = extra_action

        if graph_generator != None:
            assert isinstance(graph_generator,
                              GraphGenerator), "graph_generator must be a GraphGenerator implementation."
            self.gg = graph_generator
        else:
            # provide a default graph generator if one is not passed
            self.gg = RandomGraphGenerator(n_spins=20,
                                           edge_type=EdgeType.DISCRETE,
                                           biased=False,
                                           extra_action=(extra_action != extra_action.NONE))

        self.n_spins = self.gg.n_spins  # Total number of spins in episode
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_spins
        if extra_action != ExtraAction.NONE:
            self.n_actions += 1

        self.action_space = self.action_space(self.n_actions)
        self.observation_space = self.observation_space(self.n_spins, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.spin_basis = spin_basis

        self.memory_length = memory_length
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_spins = reversible_spins

        self.reset()

        self.score = self.calculate_score()
        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_spins = self.state[0, :]

        if init_snap != None:
            self.load_snapshot(init_snap)

        self.random_weight = np.array(np.random.randint(1, 10, size=self.n_spins))
        self.ifweight = ifweight

    def reset(self, spins=None, reset_weight=True):
        """
        Explanation here
        """
        if reset_weight:
            self.random_weight = np.array(np.random.randint(1, 10, size=self.n_spins))
        self.current_step = 0
        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
        self._reset_graph_observables()
        spinsOne = np.array([1] * self.n_spins)
        local_rewards_available = self.get_immeditate_rewards_avaialable(spinsOne)
        local_rewards_available = local_rewards_available[np.nonzero(local_rewards_available)]
        if local_rewards_available.size == 0:
            self.reset()
        else:
            self.max_local_reward_available = np.max(local_rewards_available)

        self.state = self._reset_state(spins)
        self.score = 0

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_obs_score = self.score
        self.best_spins = self.state[0, :self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        if self.memory_length is not None:
            self.score_memory = np.array([self.best_score] * self.memory_length)
            self.spins_memory = np.array([self.best_spins] * self.memory_length)
            self.idx_memory = 1

        self._reset_graph_observables()

        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer()

        return self.get_observation()

    def _reset_graph_observables(self):
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action spins of value 0.
            self.matrix_obs = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[0] + 1))
            self.matrix_obs[:-1, :-1] = self.matrix
        else:
            self.matrix_obs = self.matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action spins of value 0.
                self.bias_obs = np.concatenate((self.bias, [0]))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, spins=None):
        state = np.zeros((self.observation_space.shape[1], self.n_actions))

        if spins is None:
            if self.reversible_spins:
                state[0, :self.n_spins] = 2 * np.random.randint(2, size=self.n_spins) - 1
            else:
                state[0, :self.n_spins] = 1
        else:
            state[0, :] = self._format_spins_to_signed(spins)

        state = state.astype('float')

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        for idx, obs in self.observables:
            if obs == Observable.IMMEDIATE_REWARD_AVAILABLE:
                state[idx, :self.n_spins] = self.get_immeditate_rewards_avaialable(
                    spins=state[0, :self.n_spins]) / self.max_local_reward_available
            elif obs == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable(spins=state[0, :self.n_spins])
                state[idx, :self.n_spins] = 1 - np.sum(immeditate_rewards_avaialable <= 0) / self.n_spins

        return state

    def _get_spins(self, basis=SpinBasis.SIGNED):
        spins = self.state[0, :self.n_spins]

        if basis == SpinBasis.SIGNED:
            pass
        elif basis == SpinSystemBiased:
            # convert {1,-1} --> {0,1}
            spins[0, :] = (1 - spins[0, :]) / 2
        else:
            raise NotImplementedError("Unrecognised SpinBasis")

        return spins

    def calculate_best_energy(self):
        if self.n_spins <= 10:
            # Generally, for small systems the time taken to start multiple processes is not worth it.
            res = self.calculate_best_brute()

        else:
            # Start up processing pool
            n_cpu = int(mp.cpu_count()) / 2

            pool = mp.Pool(mp.cpu_count())

            # Split up state trials across the number of cpus
            iMax = 2 ** (self.n_spins)
            args = np.round(np.linspace(0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
            arg_pairs = [list(args) for args in zip(args, args[1:])]

            try:
                res = pool.starmap(self._calc_over_range, arg_pairs)
                # Return the best solution,
                idx_best = np.argmin([e for e, s in res])
                res = res[idx_best]
            except Exception as e:
                # Falling back to single-thread implementation.
                # res = self.calculate_best_brute()
                res = self._calc_over_range(0, 2 ** (self.n_spins))
            finally:
                # No matter what happens, make sure we tidy up after outselves.
                pool.close()

            if self.spin_basis == SpinBasis.BINARY:
                # convert {1,-1} --> {0,1}
                best_score, best_spins = res
                best_spins = (1 - best_spins) / 2
                res = best_score, best_spins

            if self.optimisation_target == OptimisationTarget.DSP:
                best_energy, best_spins = res
                best_cut = self.calculate_dsp(best_spins)
                res = best_cut, best_spins
            else:
                raise NotImplementedError()

        return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action, random_weight, adj_mask=False):
        done = False
        rew = 0  # Default reward to zero.
        randomised_spins = False
        self.current_step += 1

        if self.current_step > self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError
        new_state = np.copy(self.state)

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################
        if action == self.n_spins:
            if self.extra_action == ExtraAction.PASS:
                delta_score = 0
            if self.extra_action == ExtraAction.RANDOMISE:
                randomised_spins = True
                random_actions = np.random.choice([1, -1], self.n_spins)
                new_state[0, :] = self.state[0, :] * random_actions
                new_score = self.calculate_score(new_state[0, :])
                delta_score = new_score - self.score
                self.score = new_score
        else:
            # Perform the action and calculate the score change.
            new_state[0, action] = -self.state[0, action]


            if self.gg.biased:
                delta_score = self._calculate_score_change(new_state[0, :self.n_spins], self.matrix, self.bias, action,
                                                           self.ifweight)
            else:
                # here
                delta_score = self._calculate_score_change(new_state[0, :self.n_spins], self.matrix, action,
                                                           random_weight, self.ifweight)
            self.score += delta_score

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and spin parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/spins to their respective buffers.                                  #
        #          - Update best observable score and spins w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and spin parameters. #
        #                                                                                           #

        if adj_mask:
            new_state[0, np.where(self.matrix[action, :])] = -1
        self.state = new_state
        immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable()

        if self.score > self.best_obs_score:
            if self.reward_signal == RewardSignal.BLS:
                rew = self.score - self.best_obs_score
            elif self.reward_signal == RewardSignal.CUSTOM_BLS:
                rew = self.score - self.best_obs_score
                rew = rew / (rew + 0.1)

        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score
        elif self.reward_signal == RewardSignal.SINGLE and done:
            rew = self.score - self.init_score

        if self.norm_rewards:
            rew /= self.n_spins

        if self.stag_punishment is not None or self.basin_reward is not None:
            visiting_new_state = self.history_buffer.update(action)
            # pass
        if self.stag_punishment is not None:
            if not visiting_new_state:
                rew -= self.stag_punishment
            # pass

        if self.basin_reward is not None:
            if np.all(immeditate_rewards_avaialable <= 0):
                # All immediate score changes are +ive <--> we are in a local minima.
                if visiting_new_state:
                    # #####TEMP####
                    # if self.reward_signal != RewardSignal.BLS or (self.score > self.best_obs_score):
                    # ####TEMP####
                    rew += self.basin_reward
            # pass

        if self.score > self.best_score:
            self.best_score = self.score
            self.best_spins = self.state[0, :self.n_spins].copy()

        if self.memory_length is not None:
            # For case of finite memory length.
            self.score_memory[self.idx_memory] = self.score
            self.spins_memory[self.idx_memory] = self.state[0, :self.n_spins]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max()
            self.best_obs_spins = self.spins_memory[self.score_memory.argmax()].copy()
        else:
            self.best_obs_score = self.best_score
            self.best_obs_spins = self.best_spins.copy()

        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the spin     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #
        #   b) Update global features in self.state (always w.r.t. best observable score/spins)     #
        #############################################################################################

        for idx, observable in self.observables:
            ### Local observables ###
            if observable == Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.state[idx, :self.n_spins] = immeditate_rewards_avaialable / self.max_local_reward_available

            elif observable == Observable.TIME_SINCE_FLIP:
                self.state[idx, :] += (1. / self.max_steps)
                if randomised_spins:
                    self.state[idx, :] = self.state[idx, :] * (random_actions > 0)
                else:
                    self.state[idx, action] = 0

            ### Global observables ###
            elif observable == Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)

            elif observable == Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.state[idx, :] = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                self.state[idx, :] = 1 - np.sum(immeditate_rewards_avaialable <= 0) / self.n_spins

            elif observable == Observable.DISTANCE_FROM_BEST_SCORE:
                self.state[idx, :] = np.abs(self.score - self.best_obs_score) / self.max_local_reward_available

            elif observable == Observable.DISTANCE_FROM_BEST_STATE:
                self.state[idx, :self.n_spins] = np.count_nonzero(
                    self.best_obs_spins[:self.n_spins] - self.state[0, :self.n_spins])

        if self.current_step == self.max_steps:
            done = True

        if not self.reversible_spins:
            if len((self.state[0, :self.n_spins] > 0).nonzero()[0]) == 0:
                done = True

        return (self.get_observation(), rew, done, None)

    def get_observation(self):
        state = self.state.copy()

        if self.spin_basis == SpinBasis.BINARY:
            state[0, :] = (1 - state[0, :]) / 2

        if self.gg.biased:
            return np.vstack((state, self.matrix_obs, self.bias_obs))
        else:
            return np.vstack((state, self.matrix_obs))

    def get_immeditate_rewards_avaialable(self, spins=None):
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target == OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1 * self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target == OptimisationTarget.DSP:
            immediate_reward_function = self._get_immeditate_dsps_avaialable_jit
        else:
            raise NotImplementedError("Optimisation target {} not recognised.".format(self.optimisation_ta))

        spins = spins.astype('float64')
        matrix = self.matrix_obs.astype('float64')
        if self.gg.biased:
            bias = self.bias.astype('float64')
            return immediate_reward_function(spins, matrix, bias)
        else:
            return immediate_reward_function(spins, matrix)

    def get_allowed_action_states(self):
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0, 1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (1, -1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis == SpinBasis.BINARY:
                return 0
            if self.spin_basis == SpinBasis.SIGNED:
                return 1

    def calculate_score(self, spins=None):
        if self.optimisation_target == OptimisationTarget.DSP:
            score = self.calculate_dsp(spins)
        else:
            raise NotImplementedError
        return score

    def _calculate_score_change(self, new_spins, matrix, action, random_weight, ifweight):
        if self.optimisation_target == OptimisationTarget.DSP:
            delta_score = self._calculate_dsp_change(new_spins, matrix, action, random_weight, ifweight)

        else:
            raise NotImplementedError
        return delta_score

    def _format_spins_to_signed(self, spins):
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception("SpinSystem is configured for binary spins ([0,1]).")
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception("SpinSystem is configured for signed spins ([-1,1]).")
        return spins



    @abstractmethod
    def calculate_dsp(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_dsp(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    # @abstractmethod
    # def _calculate_energy_change(self, new_spins, matrix, action):
    #     raise NotImplementedError

    @abstractmethod
    def _calculate_dsp_change(self, new_spins, matrix, action, random_weight, ifweight):
        raise NotImplementedError


##########
# Classes for implementing the calculation methods with/without biases.
##########
class SpinSystemUnbiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(spins, matrix)

    def calculate_dsp(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        return (1 / 4) * np.sum(np.multiply(self.matrix, 1 - np.outer(spins, spins)))

    def get_best_dsp(self):
        if self.optimisation_target == OptimisationTarget.DSP:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best DSP solve when optimisation target is set to energy.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix)

    def _calculate_dsp_change(self, new_spins, matrix, action, random_weight, ifweight):
        if ifweight:
            spins_state = deepcopy(new_spins)
            spins_state[spins_state == -1] = 0
            res = np.sum((spins_state * matrix[:, action]) * random_weight) + random_weight[action]
            return res
        else:
            return matmul(new_spins.T, matrix[:, action])  # new_spins[action] *


    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - matmul(spins.T, matmul(matrix, spins)) / 2
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins


    @staticmethod
    def _get_immeditate_dsps_avaialable_jit(spins, matrix):
        spins_state = copy(spins)
        spins_state[spins_state == -1] = 0
        return matmul(matrix, spins)


class SpinSystemBiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if type(spins) == type(None):
            spins = self._get_spins()

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')

        return self._calculate_energy_jit(spins, matrix, bias)

    def calculate_dsp(self, spins=None):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    def get_best_dsp(self):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix, bias)

    @staticmethod
    @jit(nopython=True)
    def _calculate_dsp_change(new_spins, matrix, bias, action):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_jit(spins, matrix, bias):
        return matmul(spins.T, matmul(matrix, spins)) / 2 + matmul(spins.T, bias)

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix, bias):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            current_energy = -(matmul(spins.T, matmul(matrix, spins)) / 2 + matmul(spins.T, bias))
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix, bias):
        return - (2 * spins * (matmul(matrix, spins) + bias))

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_dsps_avaialable_jit(spins, matrix, bias):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")
