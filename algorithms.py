from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import EIIE
from .utils import apply_portfolio_noise
from .utils import PVM
from .utils import ReplayBuffer
from .utils import RLDataset


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0,
        optimizer=AdamW,
        device="cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
          env: Training Environment.
          policy: Policy architecture to be used.
          policy_kwargs: Arguments to be used in the policy network.
          validation_env: Validation environment.
          batch_size: Batch size to train neural network.
          lr: policy Neural network learning rate.
          action_noise: Noise parameter (between 0 and 1) to be applied
            during training.
          optimizer: Optimizer of neural network.
          device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.optimizer = optimizer
        self.device = device
        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

    def _setup_train(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before training.

        Args:
          env: environment.
          policy: Policy architecture to be used.
          batch_size: Batch size to train neural network.
          lr: Policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.train_env = env

        # neural networks
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def train(self, episodes=100):
        """Training sequence.

        Args:
            episodes: Number of episodes to simulate.
        """
        for i in tqdm(range(1, episodes + 1)):
            obs = self.train_env.reset()  # observation
            self.train_pvm.reset()  # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = apply_portfolio_noise(
                    self.train_policy(obs_batch, last_action_batch), self.action_noise
                )
                self.train_pvm.add(action)

                # run simulation step
                next_obs, reward, done, info = self.train_env.step(action)

                # add experience to replay buffer
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.train_buffer.append(exp)

                # update policy networks
                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()

            # validation step
            if self.validation_env:
                self.test(self.validation_env)

    def _setup_test(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before testing.

        Args:
          env: Environment.
          policy: Policy architecture to be used.
          batch_size: batch size to train neural network.
          lr: policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.test_env = env

        # process None arguments
        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        # neural networks
        # define policy
        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.test_buffer = ReplayBuffer(capacity=batch_size)
        self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test(
        self, env, policy=None, online_training_period=10, lr=None, optimizer=None
    ):
        """Tests the policy with online learning.

        Args:
          env: Environment to be used in testing.
          policy: Policy architecture to be used. If None, it will use the training
            architecture.
          online_training_period: Period in which an online training will occur. To
            disable online learning, use a very big value.
          batch_size: Batch size to train neural network. If None, it will use the
            training batch size.
          lr: Policy neural network learning rate. If None, it will use the training
            learning rate
          optimizer: Optimizer of neural network. If None, it will use the training
            optimizer

        Note:
            To disable online learning, set learning rate to 0 or a very big online
            training period.
        """
        self._setup_test(env, policy, online_training_period, lr, optimizer)

        obs = self.test_env.reset()  # observation
        self.test_pvm.reset()  # reset portfolio vector memory
        done = False
        steps = 0

        while not done:
            steps += 1
            # define last_action and action and update portfolio vector memory
            last_action = self.test_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            action = self.test_policy(obs_batch, last_action_batch)
            self.test_pvm.add(action)

            # run simulation step
            next_obs, reward, done, info = self.test_env.step(action)

            # add experience to replay buffer
            exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            self.test_buffer.append(exp)

            # update policy networks
            if steps % online_training_period == 0:
                self._gradient_ascent(test=True)

            obs = next_obs

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
        """
        # get batch data from dataloader
        obs, last_actions, price_variations, trf_mu = (
            next(iter(self.test_dataloader))
            if test
            else next(iter(self.train_dataloader))
        )
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)

        # define policy loss (negative for gradient ascent)
        mu = (
            self.test_policy.mu(obs, last_actions)
            if test
            else self.train_policy.mu(obs, last_actions)
        )
        policy_loss = -torch.mean(
            torch.log(torch.sum(mu * price_variations * trf_mu, dim=1))
        )

        # update policy network
        if test:
            self.test_policy.zero_grad()
            policy_loss.backward()
            self.test_optimizer.step()
        else:
            self.train_policy.zero_grad()
            policy_loss.backward()
            self.train_optimizer.step()


import torch
import torch.nn.functional as F
from torch.optim import Adam
import copy
from collections import deque
import random


class CustomReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class TD3:
    def __init__(
        self,
        env, #gym train env
        actor_critic, #CustomGPM
        actor_critic_target,
        action_noise=0.1,
        gradient_steps=1,
        lr=1e-3,
        tau=0.005,
        gamma=0.99,
        buffer_size=1000000,
        batch_size=100,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_delay=2,
        device="cpu"
    ):
        self.env = env
        self.action_noise = action_noise
        self.n_updates = 0
        self.gradient_steps = gradient_steps
        self.actor_critic = actor_critic
        self.actor_critic_target = actor_critic_target
        self.actor_optimizer = Adam(self.actor_critic.track_actor_parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.actor_critic.track_critic1_parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.actor_critic.track_critic2_parameters(), lr=lr)
        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = CustomReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.device = device

    def select_action(self, obs):
        obs_batch = np.expand_dims(obs["state"], axis=0)
        last_action_batch = np.expand_dims(obs["last_action"], axis=0)
        action = self.actor_critic(obs_batch, last_action_batch, mode="actor") #tensor(1, portfolio_size+1)
        action = action.cpu().detach().numpy().squeeze() #np.array(protfolio_size+1,)

        action = (action + np.random.normal(0, self.action_noise, size=action.shape)).clip(self.env.action_space.low,
                                                                                           self.env.action_space.high)
        action /= np.sum(action) #portfolio weights sum equals 1
        return action

    def train(self, total_steps):
        state = self.env.reset() #state: space.Dict
        episode_reward = 0
        for _ in range(total_steps):
            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(self.replay_buffer) > self.batch_size:
                self.update_network()

            if done:
                state = self.env.reset()
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0

    def update_network(self):
        for _ in range(self.gradient_steps):
            self.n_updates += 1
            transitions = self.replay_buffer.sample(self.batch_size)
            states, last_actions, actions, rewards, next_states, _, dones = zip(*[
                (s['state'], s['last_action'], a, r, ns['state'], ns['last_action'], d) for s, a, r, ns, d in
                transitions
            ])

            states = torch.FloatTensor(states).to(self.device)
            last_actions = torch.FloatTensor(last_actions).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            with torch.no_grad():
                noise = (torch.randn_like(actions) * self.target_policy_noise).clamp(-self.target_noise_clip,
                                                                                     self.target_noise_clip)
                action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
                action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
                next_actions = (self.actor_critic_target(next_states, actions, mode="actor") + noise).clamp(action_low,
                                                                                                            action_high)
                next_actions = next_actions / next_actions.sum(dim=1, keepdim=True)  # portfolio weights sum equals 1

                # Compute the target Q values
                target_Q1 = self.actor_critic_target(next_states, next_actions, mode="critic1")
                target_Q2 = self.actor_critic_target(next_states, next_actions, mode="ciritc2")
                target_Q = rewards + ((1 - dones) * self.gamma * torch.min(target_Q1, target_Q2))

            # Get current Q estimates
            current_Q1 = self.actor_critic(states, actions, mode="critic1")
            current_Q2 = self.actor_critic(states, actions, mode="critic2")

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if self.n_updates % self.policy_delay == 0:
                actor_loss = -self.actor_critic(states, self.actor_critic(states, last_actions, mode="actor"),
                                                mode="critic1").mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

class PPO:
    def __init__(
        self,
        env,
        actor_model,
        critic_model,
        lr=1e-4,
        ent_coef=0.01,
        gamma=0.99,
        clip_epsilon=0.2,
        gae_lambda=0.95,
        buffer_size=2048,
        minibatch_size=64,
        num_episodes=10,
        ppo_epochs=2,
        device="cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.ent_coef = ent_coef,
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.num_episodes = num_episodes
        self.ppo_epochs = ppo_epochs

        self.actor = actor_model.to(device)
        self.critic = critic_model.to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.device = device
        self.replay_buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Clipping action to the valid range
        action = torch.clamp(action, min=-1.0, max=1.0).squeeze(0)

        return action.cpu().numpy(), log_prob.item(), dist.entropy().item()

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def train(self):

        states, actions, action_log_probs, rs, next_states, ds, entropies = zip(*self.replay_buffer)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        action_log_probs = torch.FloatTensor(action_log_probs).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rs).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(rs).unsqueeze(1).to(self.device)
        entropies = torch.FloatTensor(rs).unsqueeze(1).to(self.device)

        values = self.critic(states)
        next_values = self.critic(next_states).detach()

        # Calculate GAE and returns
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + (self.gamma * self.gae_lambda * advantages[t + 1] * (1 - dones[t]))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values
        dataset = TensorDataset(states, actions, action_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # PPO updates
        for _ in range(self.ppo_epochs):
            for batch in loader:
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch

                dist = self.actor(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                new_values = self.critic(batch_states).squeeze()

                # Calculate ratios
                ratios = (new_log_probs - batch_old_log_probs).exp()

                # Clipped surrogate function
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                critic_loss = F.mse_loss(new_values, batch_returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer = []
    def run(self):
        for _ in range(self.num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action, log_prob, entropy = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition((state, action, log_prob, reward, next_state, done, entropy))
                state = next_state

                if len(self.replay_buffer) >= self.buffer_size:
                    self.train()




