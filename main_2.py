import gym

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utilities import *
from models import *
# problem = "Pendulum-v0"
# model = gym.make(problem)
dt = 0.5
# num_states = model.observation_space.shape[0]
# num_actions = model.action_space.shape[0]

# upper_bound = model.action_space.high[0]
# lower_bound = model.action_space.low[0]
# state = model.reset()

# next_state, reward, done, info =model.step([0])
# print(next_state)
# Create an agent instance
from pylab import *

p = {"F_nom": 1.5,  # m3/s
     "F_dev": 0.5,  # m3/s
     "freq": 1 / (2 * pi)}

tf = 5
steps = 0.05
ttt = np.arange(0, tf + steps, steps)
controls = np.array([[1]])  # for _ in range(ttt.size-1)]])
x0 = np.array([0, 0])

m = ModelIntegration(p)
m.reset()
xt, reward, done = m.simulation(controls, tf, x0, 1)
print(m.reset())
model = ModelIntegration(p)  # model_double_intergator()

agent = ActorCriticAgent(model, network=PTACNetwork)
print(2)
# Define number of episodes to train for
num_episodes = 2000
# Create a buffer for calculating the last 100 episode average reward
scores_buffer = deque(maxlen=150)
# List to store each episode's total reward
scores = []
# List to store the average reward after each episode
avg_scores = []
# Run the training loop
for ep in range(num_episodes):
    # Save the initial state
    state = model.reset()
    t = 0
    # Reset the total reward
    total_reward = 0
    # Reset the episode terminal condition
    done = False
    if ep == num_episodes - 1:
        states = []
        uu = []
    while t < 5:  # not done:
        # if t%100==0:
        #   state = xx[ep*t+1]
        # Query the agent for an action to take in the state
        # Change the state to get previous states and deviations
        for ii in range(100):
            u = agent.get_action(np.array(state)/[4,2])
            k = 0
            for i in range(model.nu):
                if u[i]>model.u_min-0.00001 and u[i]<model.u_max+0.00001:
                   k+=1
            if k==model.nu:
                break
        # Take the action in the environment
        # next_state, reward, done, info =model.step(u)  # Change this to run with the regular funcs
        # print('before: ', u)
        # if ep >3:


        # print('after: ', u)
        # for i in range(model.nu):
        #   u[i] = 2*np.sin(u[i])
        # if u[i]<model.u_min:
        #   u[i] = model.u_min
        # elif u[i]>model.u_max:
        #   u[i] = model.u_max
        F_nom = p["F_nom"]
        F_dev = p["F_dev"]
        freq = p["freq"]

        # algebraic equations
        d = F_nom + F_dev * sin(t / freq)
        if ep == num_episodes - 1:
            uu += [u]
            states += [state]
        next_state, reward, done = m.simulation(u.reshape((-1,)), 0.01, np.array([[state[0], 0]]).reshape((-1,)), d)

        # next_state, reward, done =model.simulate(state, u, t, 0.0)# model.step(u)  # Change this to run with the regular funcs
        # next_state += 0.2*np.random.rand()
        # Train the agent with the new time step experience
        agent.train(state, u, next_state, reward, int(done))
        # Update the episode's total reward
        total_reward += reward
        # Update the current state
        state = next_state
        t += 0.01

    # Store the last episode's total reward
    scores.append(total_reward)
    # Add the total reward to the buffer for calculating average reward
    scores_buffer.append(total_reward)
    # Store the new average reward
    avg_scores.append(np.mean(scores_buffer))
    print("Episode: ", ep, "Score: ", scores[ep], "Avg reward: ", avg_scores[ep])

plt.plot(states)
print(2)