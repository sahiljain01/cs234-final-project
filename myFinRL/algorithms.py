from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import EIIE
from utils import apply_portfolio_noise
from utils import PVM
from utils import ReplayBuffer
from utils import RLDataset


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
from tqdm import tqdm
from math import log


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
        train_env, #gym train env
        test_env, #gym_test_env
        best_model_path,
        metric_file_path,
        actor_critic, #CustomGPM
        actor_critic_target,
        action_noise=0.1,
        gradient_steps=1,
        lr=1e-3,
        tau=0.005,
        gamma=0.99,
        buffer_size=640,
        batch_size=16,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_delay=2,
        device="cpu",
        early_stopping_threshold=0.005,
        tolerance=5,
        online_training_period=10
    ):
        self.train_env = train_env
        self.test_env = test_env
        self.best_model_path = best_model_path
        self.metric_file_path = metric_file_path
        self.action_noise = action_noise
        self.n_updates = 0
        self.lr = lr
        self.gradient_steps = gradient_steps
        self.actor_critic = actor_critic.to(device)
        self.actor_critic_target = actor_critic_target.to(device)
        self.actor_optimizer = Adam(self.actor_critic.track_actor_parameters(), lr=lr)
        self.critic_optimizer = Adam(self.actor_critic.track_critic_parameters(), lr=lr)
        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = CustomReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.device = device
        self.early_stopping_threshold = early_stopping_threshold
        self.tolerance = tolerance
        self.prev_test_log_return = -float('inf')
        self.no_improvement_count = 0
        self.online_training_period = online_training_period

    def select_action(self, obs):
        obs_batch = torch.from_numpy(np.expand_dims(obs["state"], axis=0)).to(self.device)
        last_action_batch = torch.from_numpy(np.expand_dims(obs["last_action"], axis=0)).to(self.device)
        action = self.actor_critic(obs_batch, last_action_batch, mode="actor") #tensor(1, portfolio_size+1)
        action = action.cpu().detach().numpy().squeeze() #np.array(protfolio_size+1,)
        action_w_noise = apply_portfolio_noise(action, epsilon=0.1)

        #action = (action + np.random.normal(0, self.action_noise, size=action.shape)).clip(self.train_env.action_space.low,
        #                                                                                   self.train_env.action_space.high)
        #normalised_action = np.exp(action) / np.sum(np.exp(action)) #portfolio weights sum equals 1
        #return normalised_action
        return action_w_noise
    def train(self, total_episodes):
        self.actor_critic.train()
        self.actor_critic_target.eval() # target network not participating in gradient descent
        best_train_results = []

        with open(self.metric_file_path, 'w') as metrics_file:
            metrics_file.write("Episodes, train_final_pv, train_log_return, test_final_pv, test_log_return\n")

            with tqdm(total=total_episodes, desc="Train: ") as pbar:
                for episode in range(total_episodes):
                    print("Start episode ", episode + 1)
                    state = self.train_env.reset()  # state: space.Dict
                    #episode_reward = 0
                    done = False
                    while not done:
                        action = self.select_action(state)

                        next_state, reward, done, _ = self.train_env.step(action)
                        self.replay_buffer.add((state, action, reward, next_state, done))
                        state = next_state
                        #episode_reward += reward

                        if len(self.replay_buffer) > self.batch_size:
                            self.update_network()
                            print("updating network")

                    train_final_portfolio_value = self.train_env._portfolio_value
                    #train_final_portfolio_weights = self.train_env._final_weights[-1]
                    train_log_return = log(self.train_env._portfolio_value / self.train_env._asset_memory["final"][0])

                    print("Done!")
                    #print(f"Episode reward: {episode_reward}")
                    print(f"Training log return: {train_log_return}")
                    print("Start validation/testing")

                    self.test()
                    test_final_portfolio_value = self.test_env._portfolio_value
                    #test_final_portfolio_weights = self.test_env._final_weights[-1]
                    test_log_return = log(self.test_env._portfolio_value / self.test_env._asset_memory["final"][0])
                    print(f"Validation log return: {test_log_return}")

                    # Log the results
                    print(f"TD3_GPM_: Episodes {episode + 1}: train_log_return :: {train_log_return :.3f}, test_log_return :: {test_log_return :.3f}")
                    metrics_file.write(
                        f"{episode + 1},{train_final_portfolio_value:.3f},{train_log_return:.3f},{test_final_portfolio_value:.3f},{test_log_return:.3f}\n")

                    # Early stopping check
                    if test_log_return - self.prev_test_log_return < self.early_stopping_threshold:
                        self.no_improvement_count += 1
                        print("no improvement")
                        if self.no_improvement_count >= self.tolerance:
                            print("Early stopping criteria met. Training stopped.")
                            break
                    else:
                        print("best model found!")
                        self.no_improvement_count = 0
                        self.prev_test_log_return = test_log_return
                        best_train_results = self.train_env._asset_memory["final"]
                        best_test_results = self.test_env._asset_memory["final"]
                        torch.save(self.actor_critic.state_dict(), self.best_model_path)
                        print(f"Model saved to {self.best_model_path}")

                    pbar.update(1)

        return best_train_results, best_test_results

    def update_network(self):
        for _ in range(self.gradient_steps):
            self.n_updates += 1
            transitions = self.replay_buffer.sample(self.batch_size)
            states, last_actions, actions, rewards, next_states, _, dones = zip(*[
                (s['state'], s['last_action'], a, r, ns['state'], ns['last_action'], d) for s, a, r, ns, d in
                transitions
            ])

            states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
            last_actions = torch.tensor(np.array(last_actions), dtype=torch.float).to(self.device)
            actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
            dones = torch.tensor(np.array(dones), dtype=torch.float).unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_actions = self.actor_critic_target(next_states, actions, mode="actor")
                next_actions_w_noise = []
                for i in range(next_actions.shape[0]):
                    next_action = next_actions[i].cpu().numpy()
                    next_action = apply_portfolio_noise(next_action, epsilon=self.target_policy_noise)
                    next_actions_w_noise.append(next_action)
                next_actions_w_noise = torch.tensor(np.array(next_actions_w_noise), dtype=torch.float).to(self.device)

                # Compute the target Q values
                target_Q1 = self.actor_critic_target(next_states, next_actions_w_noise, mode="critic1")
                target_Q2 = self.actor_critic_target(next_states, next_actions_w_noise, mode="ciritc2")
                target_Q = rewards + ((1 - dones) * self.gamma * torch.min(target_Q1, target_Q2))

            # Get current Q estimates
            current_Q1 = self.actor_critic(states, actions, mode="critic1")
            current_Q2 = self.actor_critic(states, actions, mode="critic2")

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.n_updates % self.policy_delay == 0:
                actor_loss = -self.actor_critic(states, self.actor_critic(states, last_actions, mode="actor"),
                                                mode="critic1").mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def test(self):
        self.test_policy_net = copy.deepcopy(self.actor_critic).to(self.device)
        self.test_policy_net.eval()
        self.test_optimizer = Adam(self.test_policy_net.track_actor_parameters(), lr=self.lr)

        # replay buffer
        self.test_buffer = ReplayBuffer(capacity=self.batch_size)

        obs = self.test_env.reset()  # observation
        done = False
        steps = 0

        while not done:
            steps += 1
            # define last_action and action and update portfolio vector memory
            last_action = obs["last_action"]
            obs = obs["state"]
            obs_batch = torch.from_numpy(np.expand_dims(obs, axis=0)).to(self.device)
            last_action_batch = torch.from_numpy(np.expand_dims(last_action, axis=0)).to(self.device)

            action = self.test_policy_net(obs_batch, last_action_batch, mode="actor")
            action = action.cpu().detach().numpy().squeeze()

            # run simulation step
            next_obs, reward, done, info = self.test_env.step(action)

            # add experience to replay buffer
            exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            self.test_buffer.append(exp)

            # update policy networks
            if steps % self.online_training_period == 0:
                print("update policy net when testing")
                exps = self.test_buffer.sample()
                obs, last_actions, price_variations, trf_mu = zip(*exps)

                obs = torch.tensor(np.array(obs), dtype=torch.float).to(self.device)
                last_actions = torch.tensor(np.array(last_actions), dtype=torch.float).to(self.device)
                price_variations = torch.tensor(np.array(price_variations), dtype=torch.float).to(self.device)
                trf_mu = torch.tensor(np.array(trf_mu), dtype=torch.float).unsqueeze(1).to(self.device)

                # define policy loss
                weights = self.test_policy_net(obs, last_actions, mode="actor")

                policy_loss = -torch.mean(
                    torch.log(torch.sum(weights * price_variations * trf_mu, dim=1))
                )

                # update policy network
                self.test_policy_net.zero_grad()
                policy_loss.backward()
                self.test_optimizer.step()

            obs = next_obs


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd

torch.autograd.set_detect_anomaly(True)

class PPO:
    def __init__(
        self,
        train_env,
        test_env,
        best_model_path,
        metric_file_path,
        actor_critic,
        lr=1e-3,
        ent_coef=0.01,
        gamma=0.99,
        clip_epsilon=0.2,
        gae_lambda=0.95,
        buffer_size=100,
        minibatch_size=16,
        num_episodes=10,
        ppo_epochs=4,
        device="cpu",
        early_stopping_threshold=0.005,
        tolerance=5,
        online_training_period=10
    ):
        self.train_env = train_env
        self.test_env = test_env
        self.best_model_path = best_model_path
        self.metric_file_path = metric_file_path

        self.gamma = gamma
        self.ent_coef = ent_coef,
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs
        self.num_episodes = num_episodes
        self.actor_critic = actor_critic.to(device)
        self.log_std = nn.Parameter(torch.zeros(self.train_env.portfolio_size + 1)).to(device)
        self.actor_optimizer = Adam(self.actor_critic.track_actor_parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.actor_critic.track_critic1_parameters(), lr=lr)
        self.lr=lr

        self.device = device
        self.train_replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.early_stopping_threshold = early_stopping_threshold
        self.tolerance = tolerance
        self.prev_test_log_return = -float('inf')
        self.no_improvement_count = 0
        self.online_training_period = online_training_period

    def select_action(self, obs):
        obs_batch = torch.from_numpy(np.expand_dims(obs["state"], axis=0)).to(self.device)
        last_action_batch = torch.from_numpy(np.expand_dims(obs["last_action"], axis=0)).to(self.device)

        with torch.no_grad():
            mean = self.actor_critic(obs_batch, last_action_batch, mode="actor_mean")
            std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum() # [1, portfolio_size+1] -> []
        entropy = dist.entropy().sum()

        # Clipping action to the valid range
        action = torch.softmax(torch.clamp(action, min=-1.0, max=1.0), dim=1).squeeze(0)

        return action.cpu().numpy(), log_prob.item(), entropy.item()

    def store_transition(self, transition):
        self.train_replay_buffer.append(transition)

    def train(self):
        states, last_actions, actions, action_log_probs, rs, next_states, _, ds, entropies = zip(*self.train_replay_buffer.sample())

        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        last_actions = torch.tensor(np.array(last_actions), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
        action_log_probs = torch.tensor(np.array(action_log_probs), dtype=torch.float).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rs), dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(ds), dtype=torch.float).unsqueeze(1).to(self.device)
        entropies = torch.tensor(np.array(entropies), dtype=torch.float).unsqueeze(1).to(self.device)

        values = self.actor_critic(states, last_actions, mode="value").detach() #[N, 1]
        next_values = self.actor_critic(next_states, actions, mode="value").detach()

        # Calculate GAE and returns
        deltas = rewards + self.gamma * next_values * (1 - dones) - values #[N, 1]
        advantages = torch.zeros_like(deltas)

        for t in reversed(range(len(deltas))):
            if t == len(deltas) - 1:
                advantages[t] = deltas[t]
            else:
                advantages[t] = deltas[t] + (self.gamma * self.gae_lambda * advantages[t + 1] * (1 - dones[t]))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values

        dataset = TensorDataset(states, last_actions, actions, action_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # PPO updates
        for _ in range(self.ppo_epochs):
            for batch in loader:
                batch_states, batch_last_actions, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch

                mean = self.actor_critic(batch_states, batch_last_actions, mode="actor_mean")
                std = torch.exp(self.log_std)
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()

                # Calculate ratios
                ratios = (new_log_probs - batch_old_log_probs).exp()

                # Clipped surrogate function
                surr1 = ratios * batch_advantages.detach()
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean() - torch.tensor(self.ent_coef, device=self.device) * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                critic = self.actor_critic(batch_states, batch_actions, mode="critic1")
                critic_loss = F.mse_loss(batch_returns, critic)

                self.critic1_optimizer.zero_grad()
                critic_loss.backward()
                self.critic1_optimizer.step()

    def run(self):
        best_train_results = []

        with open(self.metric_file_path, 'w') as metrics_file:
            metrics_file.write("Episodes, train_final_pv, train_log_return, test_final_pv, test_log_return\n")

            with tqdm(total=self.num_episodes, desc="Train: ") as pbar:
                for episode in range(self.num_episodes):
                    self.actor_critic.train()
                    print("Start episode ", episode + 1)
                    state = self.train_env.reset()
                    done = False
                    while not done:
                        action, log_prob, entropy = self.select_action(state)
                        next_state, reward, done, _ = self.train_env.step(action)
                        self.store_transition((state["state"], state["last_action"], action, log_prob, reward,
                                               next_state["state"], next_state["last_action"], done, entropy))
                        state = next_state

                        if len(self.train_replay_buffer) >= self.buffer_size:
                            print("start train")
                            self.train()

                    print("start train")
                    self.train()

                    train_final_portfolio_value = self.train_env._portfolio_value
                    # train_final_portfolio_weights = self.train_env._final_weights[-1]
                    train_log_return = log(self.train_env._portfolio_value / self.train_env._asset_memory["final"][0])

                    print("Done!")
                    # print(f"Episode reward: {episode_reward}")
                    print(f"Training log return: {train_log_return}")
                    print("Start validation/testing")

                    self.test()
                    test_final_portfolio_value = self.test_env._portfolio_value
                    # test_final_portfolio_weights = self.test_env._final_weights[-1]
                    test_log_return = log(self.test_env._portfolio_value / self.test_env._asset_memory["final"][0])
                    print(f"Validation log return: {test_log_return}")

                    # Log the results
                    print(
                        f"PPO_GPM_: Episodes {episode + 1}: train_log_return :: {train_log_return :.3f}, test_log_return :: {test_log_return :.3f}")
                    metrics_file.write(
                        f"{episode + 1},{train_final_portfolio_value:.3f},{train_log_return:.3f},{test_final_portfolio_value:.3f},{test_log_return:.3f}\n")

                    # Early stopping check
                    if test_log_return - self.prev_test_log_return < self.early_stopping_threshold:
                        self.no_improvement_count += 1
                        print("no improvement")
                        if self.no_improvement_count >= self.tolerance:
                            print("Early stopping criteria met. Training stopped.")
                            break
                    else:
                        print("best model found!")
                        self.no_improvement_count = 0
                        self.prev_test_log_return = test_log_return
                        best_train_results = self.train_env._asset_memory["final"]
                        best_test_results = self.test_env._asset_memory["final"]
                        torch.save(self.actor_critic.state_dict(), self.best_model_path)
                        print(f"Model saved to {self.best_model_path}")

                    pbar.update(1)

        return best_train_results, best_test_results

    def test(self):
        obs = self.test_env.reset()  # observation
        done = False
        steps = 0

        while not done:
            steps += 1
            # define last_action and action and update portfolio vector memory
            last_action = obs["last_action"]
            obs = obs["state"]
            obs_batch = torch.from_numpy(np.expand_dims(obs, axis=0)).to(self.device)
            last_action_batch = torch.from_numpy(np.expand_dims(last_action, axis=0)).to(self.device)

            with torch.no_grad():
                mean = self.actor_critic(obs_batch, last_action_batch, mode="actor_mean")
                std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.softmax(torch.clamp(action, min=-1.0, max=1.0), dim=1).squeeze(0)
            action = action.cpu().numpy()

            # run simulation step
            next_obs, reward, done, info = self.test_env.step(action)

            obs = next_obs





