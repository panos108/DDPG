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
        m      = torch.nn.Tanh()#0.01)
        layer1 = m(self.layer1(state))
        # layer2 =m(self.layer2(layer1))
        layer3 = m(self.layer3(layer1))
        action = (self.action(layer3))
        return self.action_low + (self.action_high - self.action_low) * torch.sigmoid(action)


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net_state = nn.Linear(state_size, 100)
        self.net_action = nn.Linear(action_size, 100)
        self.net_layer = nn.Linear(200, 100)
        self.q_value = nn.Linear(100, 1)

    def forward(self, state, action):
        net_state = F.leaky_relu(self.net_state(state))
        net_action = F.leaky_relu(self.net_action(action))
        net_state_action = torch.cat([net_state, net_action], dim=1)
        net_layer = F.leaky_relu(self.net_layer(net_state_action))
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

    def update_model(self, state, action, next_state, reward, done, gamma=0.99):
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
    def __init__(self, size, scale, mu=0.0, sigma=0.05, theta=0.15, decay=0.99):
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

    def sample(self,sigma=0.2):
        #self.sigma = sigma
        sample = self.theta * (self.mu - self.noise) + self.sigma * np.random.randn(self.size)
        self.noise = sample * self.scale
        return self.noise


class ActorCriticAgent():
    # Initializing the agent and the model for selecting actions
    def __init__(self, model, network=PTACNetwork):
        # The number of state values in the state vector
        state_size = model.size_states#np.prod(model.observation_space.shape)#
        # The number of action indices to select from
        action_size = model.nu#np.prod(model.action_space.shape) #
        self.action_size = action_size
        # The continuous range of the actions
        action_range = [model.u_min, model.u_max]# [model.action_space.low, model.action_space.high] #
        # Defining the q network to use for modeling the Bellman equation
        self.q_network = network(state_size, action_size, action_range)
        # Defining the replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(50000)
        # Initializing the epsilon value to 1.0 for initial exploration
        self.noise_process = OUNoise(action_size, action_range[1] - action_range[0])
        self.action_range   = action_range

    # Function for getting an action to take in the given state
    def get_action(self, state):#, ep):
        # Get the action from the network and add noise to it
        #sigma = 0.2*np.exp(-0.01*ep)
        self.noise_process = OUNoise(self.action_size, self.action_range[1] - self.action_range[0])
        return self.q_network.get_action([state])[0] + self.noise_process.sample()#, self.action_range[0], self.action_range[1])

    def get__deterministic_action(self, state):
        # Get the action from the network and add noise to it
        return np.clip(self.q_network.get_action([state])[0],
                       self.action_range[0], self.action_range[1])# + self.noise_process.sample()

    # Function for training the agent at each time step
    def train(self, state, action, next_state, reward, done, batch_size=128*2):
        # First add the experience to the replay buffer
        self.replay_buffer.add((state, action, next_state, reward, done))
        # Sample a batch of each experience type from the replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        # Train the model with the q target
        self.q_network.update_model(states, actions, next_states, rewards, dones)
        # Decrease epsilon after each episode
        if done: self.noise_process.reset()



class cosntract_history:
    def __init__(self, model, N, store_u = True, set_point0 = 0.):
        #Define self vars
        self.model   = model           # The model defined in terms of casadi
        self.N       = N               # Number of past data
        self.store_u = store_u
        self.nx      = model.nx
        self.nu      = model.nu
        self.u_min   = model.u_min
        self.u_max   = model.u_max
        state_0, e_sp0 = model.reset(set_point0)
        # initialize history
        history_x = np.array([*state_0]*N).reshape((-1,1))
        history_sp = np.array([*e_sp0]*N).reshape((-1,1))

        if store_u:                  # If u are stored as history (simple RNN structure)
            history_u = np.array([0]*N*model.nu).reshape((-1,1))
            self.history = np.vstack((history_x,history_sp,history_u))
            self.size_states = N * (model.nu + model.nx + model.nsp)
        else:
            self.history = np.vstack((history_x,history_sp))
            self.size_states = N * (model.nx+model.nsp)

        self.history = self.history.reshape((-1,))
        # start counting the past values
        self.past = 1


    def append_history(self, new_state, u, e_sp):

        if self.store_u:
            n = self.model.nx+self.model.nu + self.model.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp, u))
            self.history[:n] = aug_states

        else:
            n = self.model.nx+ self.model.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp))

            self.history[:n] = aug_states
        self.past +=1

        return self.history