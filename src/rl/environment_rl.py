import gym
import numpy as np
import sys
import os
#utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
#sys.path.append(utils_path)
#from functions import *

import sys
sys.path.insert(1, '..')
import os
parallel_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parallel_folder_path)

from utils.functions import *    

class QuantumEnv(gym.Env):
    def __init__(self, circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits):
        super(QuantumEnv, self).__init__()
        self.circuit_fn = circuit_fn
        self.ansatz = ansatz
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.H = H
        self.num_qubits = num_qubits
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=2*np.pi, shape=(self.n_thetas,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=2*np.pi, shape=(self.n_thetas,), dtype=np.float32)
        self.state = np.random.uniform(0, 2 * np.pi, (self.n_thetas,))
        self.best_value = float('inf')
        self.best_thetas = np.zeros(self.n_thetas)
        self.best_log = {'thetas':[], 'eigenvalue':[]}
        self.states = []
        self.rewards = []
        self.done = False

    def reset(self):
        self.state = np.random.uniform(0, 2 * np.pi, (self.n_thetas,))
        self.done = False
        return self.state

    def step(self, action):
        self.state = np.clip(action, 0.0, 2 * np.pi)
        self.states.append(self.state)
        _, new_eigenvalue = get_expectation_values_hamiltonian(self.circuit_fn, self.ansatz, self.state[np.newaxis,:], self.initial_state, self.H)

        reward = new_eigenvalue

        self.rewards.append(reward)

        if len(self.rewards) > 1:
            min_r = np.min(self.rewards)
            max_r = np.max(self.rewards)
            if max_r > min_r:  # Avoid division by zero
                norm_reward = (reward - min_r) / (max_r - min_r)
            else:
                norm_reward = 0  # If all rewards are the same, normalize to 0
        else:
            norm_reward = 0  # First step normalization

        if reward < self.best_value:
            self.best_value = reward
            self.best_thetas = self.state

        self.state = np.array([norm_reward])          
        self.done = True  # Single-step optimisation
        self.best_log['eigenvalue'].append(self.best_value)
        self.best_log['thetas'].append(self.best_thetas)
        return self.state, reward, self.done, {}

    def render(self, mode='human', close=False):
        pass