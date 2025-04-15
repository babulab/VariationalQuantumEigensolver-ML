import gym
from stable_baselines3 import TD3, PPO, DDPG 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from functions import *
    
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=[64, 64], activation_fn=nn.ReLU):
        super(CustomMLP, self).__init__(observation_space, features_dim=net_arch[-1])  # Output dim is the last layer's size

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        layers = []
        input_dim = observation_space.shape[0]  # The number of inputs must match the observation space dimensions
        for output_dim in net_arch:
            layers.append(nn.Linear(input_dim, output_dim))  # Match input_dim to the previous layer's output
            layers.append(self.activation_fn())
            input_dim = output_dim  # Update input_dim to the current layer's output_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomMLP,  # Using the custom MLP
        )


class CustomMLPExtractor(nn.Module):
    def __init__(self, feature_dim: int):
        super(CustomMLPExtractor, self).__init__()
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.latent_dim_pi = 32
        self.latent_dim_vf = 32 
    def forward(self, features: th.Tensor):
        # Extract latent representations for policy and value networks
        latent_pi = self.policy_net(features)
        latent_vf = self.value_net(features)
        return latent_pi, latent_vf

    def forward_actor(self, features: th.Tensor):
        return self.policy_net(features)

    # Define the forward method for the critic (value)
    def forward_critic(self, features: th.Tensor):
        return self.value_net(features)
    

class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        # Define custom extractor to handle policy and value latents
        self.mlp_extractor = CustomMLPExtractor(self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        value = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.value_losses = []
        self.entropies = []
        self.explained_variances = []
        self.total_loss = []

    def _on_step(self) -> bool:
        # Log episode reward
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.locals['rewards'][0])
        # Access the model for more metrics
        model = self.model
        self.value_losses.append(model.logger.name_to_value.get('train/value_loss', np.nan))
        self.entropies.append(model.logger.name_to_value.get('train/entropy_loss', np.nan))
        self.explained_variances.append(model.logger.name_to_value.get('rollout/explained_variance', np.nan))
        self.total_loss.append(model.logger.name_to_value.get('train/loss', np.nan))

        return True

 # Define the environment
class OptimisationEnv(gym.Env):
    def __init__(self, target_function, num_thetas):
        super(OptimisationEnv, self).__init__()
        
        self.target_function = target_function
        self.num_angles = num_thetas
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=2*pi, shape=(self.num_angles,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=2*pi, shape=(self.num_angles,), dtype=np.float32)
        self.state = np.random.uniform(0, 2 * np.pi, (self.num_angles,))
        self.done = False
        self.best_value = float('inf')
        self.best_thetas = np.zeros(self.num_angles)
        self.history = []
        self.states = []
        self.rewards = []

    def reset(self):
        self.state =  np.random.uniform(0, 2 * np.pi, (self.num_angles,))
        self.done = False
        return self.state

    def step(self, action):
        self.state = np.clip(action, 0.0, 2 * np.pi)
        self.states.append(self.state)

        reward = self.target_function(self.state) #current_value
        self.rewards.append(reward)
        if reward < self.best_value:
            self.best_value = reward
            self.best_angles = self.state

        self.state = np.array([reward])
        self.done = True  # End the episode after one step
        self.history.append(self.best_value)

        return self.state, reward, self.done, {}

    def render(self, mode='human', close=False):
        pass

