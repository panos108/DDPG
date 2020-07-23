import numpy as np


class model_simple():

    def __init__(self):
        print('Initialize model')
        self.u_min, self.u_max, self.nu, self.nx, self.x0 = self.specifications()

    def specifications(self):
        nu = 1
        nx = 1
        u_min = np.double(np.array([0.]))  # lower bound of inputs
        u_max = np.double(np.array([1.])) # upper bound of inputs
        x0    = 1
        return u_min, u_max, nu, nx, x0

    def simulate(self, x0, u, t, ref):
        done = False
        next_state = x0*0.99 + u
        reward = -np.abs(next_state-ref)
        if t>=20 or -reward<=1e-5:
            done = True
        return next_state, reward, done

    def reset(self):
        return self.x0