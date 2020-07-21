import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_range):
        super().__init__()
        self.action_low, self.action_high = torch.from_numpy(np.array(action_range))
        self.layer1 = nn.Linear(state_size, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.action = nn.Linear(100, action_size)

    def forward(self, state):
        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))
        layer3 = F.relu(self.layer3(layer1))
        action = torch.tanh(self.action(layer3))
        return self.action_low + (self.action_high - self.action_low) * action


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net_state = nn.Linear(state_size, 100)
        self.net_action = nn.Linear(action_size, 100)
        self.net_layer = nn.Linear(200, 100)
        self.q_value = nn.Linear(100, 1)

    def forward(self, state, action):
        net_state = F.relu(self.net_state(state))
        net_action = F.relu(self.net_action(action))
        net_state_action = torch.cat([net_state, net_action], dim=1)
        net_layer = F.relu(self.net_layer(net_state_action))
        q_value = self.q_value(net_layer)
        return q_value


class PTACNetwork():
    def __init__(self, state_size, action_size, action_range):
        self.actor_local = Actor(state_size, action_size, action_range)
        self.actor_target = Actor(state_size, action_size, action_range)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=0.002)

        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.from_numpy(np.array(state)).float()
        action = self.actor_local(state).detach().numpy()
        return action

    def update_model(self, state, action, next_state, reward, done, gamma=0.97):
        states = torch.from_numpy(np.vstack(state)).float()
        actions = torch.from_numpy(np.vstack(action)).float()
        next_states = torch.from_numpy(np.vstack(next_state)).float()
        rewards = torch.from_numpy(np.vstack(reward)).float()
        dones = torch.from_numpy(np.vstack(done)).float()
        next_actions = self.actor_local(next_states)

        q_targets = rewards + gamma * self.critic_target(next_states, next_actions).detach() * (1 - dones)
        critic_loss = F.mse_loss(self.critic_local(states, actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q_baseline = self.critic_local(states, actions).detach()
        actor_gain = -(self.critic_local(states, self.actor_local(states)) - q_baseline)
        self.actor_optimizer.zero_grad()
        actor_gain.mean().backward()
        self.actor_optimizer.step()

        self.soft_copy(self.actor_local, self.actor_target)
        self.soft_copy(self.critic_local, self.critic_target)

    def soft_copy(self, local, target, tau=0.005):
        for t, l in zip(target.parameters(), local.parameters()):
            t.data.copy_(t.data + tau * (l.data - t.data))


class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))


class OUNoise():
    def __init__(self, size, scale, mu=0.0, sigma=0.4, theta=0.15, decay=0.99):
        self.noise = np.zeros(size)
        self.size = size
        self.scale = scale
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.decay = decay

    def reset(self):
        self.noise = np.zeros(self.size)
        self.scale *= self.decay

    def sample(self):
        sample = self.theta * (self.mu - self.noise) + self.sigma * np.random.randn(self.size)
        self.noise = sample * self.scale
        return self.noise


class ActorCriticAgent():
    # Initializing the agent and the model for selecting actions
    def __init__(self, env, network=PTACNetwork):
        # The number of state values in the state vector
        state_size = np.prod(env.observation_space.shape)
        # The number of action indices to select from
        action_size = np.prod(env.action_space.shape)
        # The continuous range of the actions
        action_range = [env.action_space.low, env.action_space.high]
        # Defining the q network to use for modeling the Bellman equation
        self.q_network = network(state_size, action_size, action_range)
        # Defining the replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(100000)
        # Initializing the epsilon value to 1.0 for initial exploration
        self.noise_process = OUNoise(action_size, action_range[1] - action_range[0])

    # Function for getting an action to take in the given state
    def get_action(self, state):
        # Get the action from the network and add noise to it
        return self.q_network.get_action([state])[0] + self.noise_process.sample()

    # Function for training the agent at each time step
    def train(self, state, action, next_state, reward, done, batch_size=100):
        # First add the experience to the replay buffer
        self.replay_buffer.add((state, action, next_state, reward, done))
        # Sample a batch of each experience type from the replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        # Train the model with the q target
        self.q_network.update_model(states, actions, next_states, rewards, dones)
        # Decrease epsilon after each episode
        if done: self.noise_process.reset()
