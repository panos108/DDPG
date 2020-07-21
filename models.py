import numpy as np


class model_simple():

    def __init__(self):
        print('Initialize model')
    def size_inputs_states(self):
    def bounds_inputs(self):
    def model(self, x0, u, t, ref):
        done = False
        next_state = x0*1.1 + u
        reward = np.abs(next_state-ref)
        if t<1e5 or reward<1e-5:
            done = True
        return next_state, reward, done