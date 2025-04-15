import numpy as np
import gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from functions import get_real_samples
from environment_rl import QuantumEnv
from models_rl import CustomActorCriticPolicy, CustomTD3Policy
from stable_baselines3.common.noise import NormalActionNoise


class RLQuantumOptimiserPPO:
    def __init__(self, circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits, hyperparams):
        self.env = DummyVecEnv([lambda: QuantumEnv(circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits)])
        self.hyperparams = hyperparams.copy()
        self.timesteps = self.hyperparams.pop('total_timesteps', 200)
        self.model = PPO(CustomActorCriticPolicy, self.env, **self.hyperparams)
        self.info = {}
    def optimise(self):
        self.model.learn(total_timesteps=self.timesteps)
        self.info = {'states_clipped':self.env.envs[0].states,
                    'best_eigenvalue':self.env.envs[0].best_value,
                    'best_log':self.env.envs[0].best_log}

        return self.env.envs[0].best_thetas, self.env.envs[0].best_value, self.info


class RLQuantumOptimiserTD3:
    def __init__(self, circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits, hyperparams):
        self.env = DummyVecEnv([lambda: QuantumEnv(circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits)])
        self.hyperparams = hyperparams.copy()
        self.timesteps = self.hyperparams.pop('total_timesteps', 200)
        self.noise_std = self.hyperparams.pop("noise_std", 0.01) 

        n_actions = self.env.action_space.shape[-1]

        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.noise_std* np.ones(n_actions))
        self.hyperparams['action_noise'] = action_noise
        self.model = TD3(CustomTD3Policy, self.env, **self.hyperparams)

    def optimise(self):
        self.model.learn(total_timesteps=self.timesteps)
        self.info = {'states_clipped':self.env.envs[0].states, #params thetas
                    'best_eigenvalue':self.env.envs[0].best_value,
                    'best_log':self.env.envs[0].best_log}

        return self.env.envs[0].best_thetas, self.env.envs[0].best_value, self.info

    
