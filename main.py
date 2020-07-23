import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import model_simple
from utilities import *

problem = "Pendulum-v0"
model = gym.make(problem)
#model = model_simple()
num_states = model.observation_space.shape[0]
num_actions = model.action_space.shape[0]

upper_bound = model.action_space.high[0]
lower_bound = model.action_space.low[0]


# Create an agent instance
agent = ActorCriticAgent(model, network=PTACNetwork)
# Define number of episodes to train for
num_episodes = 200
# Create a buffer for calculating the last 100 episode average reward
scores_buffer = deque(maxlen=100)
# List to store each episode's total reward
scores = []
# List to store the average reward after each episode
avg_scores = []

# Run the training loop
for ep in range(num_episodes):
    # Save the initial state
    state = model.reset()
    t     = 0
    # Reset the total reward
    total_reward = 0
    # Reset the episode terminal condition
    done = False

    while not done:
        # Query the agent for an action to take in the state
        # Change the state to get previous states and deviations
        u = agent.get_action(state)
        # Take the action in the environment
        next_state, reward, done, info = model.step(u)  # model.simulate(state, u, t, 0.)#Change this to run with the regular funcs
        #next_state += 0.2*np.random.rand()
        # Train the agent with the new time step experience
        agent.train(state, u, next_state, reward, int(done))
        # Update the episode's total reward
        total_reward += reward
        # Update the current state
        state = next_state
        t    += 1

    # Store the last episode's total reward
    scores.append(total_reward)
    # Add the total reward to the buffer for calculating average reward
    scores_buffer.append(total_reward)
    # Store the new average reward
    avg_scores.append(np.mean(scores_buffer))
    print("Episode: ", ep , "Score: ", scores[ep], "Avg reward: ", avg_scores[ep])


print(2)