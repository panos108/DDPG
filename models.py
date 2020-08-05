import numpy as np
import scipy.integrate as scp


class model_simple():

    def __init__(self):
        print('Initialize model')
        self.u_min, self.u_max, self.nu, self.nx, self.x0 = self.specifications()

    def specifications(self):
        nu = 1
        nx = 1
        u_min = np.double(np.array([-1.]))  # lower bound of inputs
        u_max = np.double(np.array([1.]))  # upper bound of inputs
        x0 = 1
        return u_min, u_max, nu, nx, x0

    def simulate(self, x0, u, t, ref):
        done = False
        next_state = x0 * 1.1 + u
        reward = -abs(next_state - ref) - 0.01 * abs(u)
        if t >= 20 or -reward <= 1e-5:
            done = True
        return next_state, reward, done

    def reset(self):
        return self.x0


class model_double_intergator():

    def __init__(self):
        print('Initialize model')
        self.u_min, self.u_max, self.nu, self.nx, self.x0 = self.specifications()

    def specifications(self):
        nu = 1
        nx = 2
        u_min = np.double(np.array([-5.]))  # lower bound of inputs
        u_max = np.double(np.array([5.]))  # upper bound of inputs
        x0 = [5, -5]
        return u_min, u_max, nu, nx, x0

    def simulate(self, x0, u, t, ref):
        done = False
        x0 = np.array(x0).reshape((self.nx, 1))
        A = np.array([[1, 0.1], [-0.2, 0.95]])
        B = np.array([[0], [1]])
        next_state = A @ x0 + B * u
        reward = -abs(next_state[0] - ref) - abs(next_state[1] - ref) - 0.01 * (u) ** 2
        if t >= 50:  # or -reward<=1e-5:
            done = True
        return next_state.reshape(self.nx, ), reward, done

    def reset(self):
        return self.x0


class ModelIntegration:
    '''
    This files integrates the model.
    model: this is were the model should be changed
    '''

    # --- initializing model --- #
    def __init__(self, parameters):
        # Object variable definitions
        self.parameters = parameters
        self.u_min = np.double(np.array([0.]))  # lower bound of inputs
        self.u_max = np.double(np.array([1.]))  # upper bound of inputs
        self.nx, self.nu, self.x_init = 2, 1, np.array([-1., 0.])

    # --- dynamic model definition --- #
    def model(self, t, state):
        # internal definitions
        params = self.parameters
        u = self.u

        # define controls here
        a = u[0]

        # state vector
        V = state[0]
        tank_t = state[1]

        # parameters

        self.F_out = a * V

        # differential equations
        dV = self.F_in - self.F_out
        dtank_t = 1

        return np.array([dV, dtank_t], dtype='float64')

    # --- simulation --- #
    def reset(self):
        return self.x_init

    def simulation(self, controls, tf, x0, d):
        '''
        u shape -> (u_{dim},steps)
        '''

        # external definitions
        self.x0, self.tf, self.F_in = 2*x0+2, tf, d*0.5 + 1.5

        # internal definitions
        steps, model, params = 20, self.model, self.parameters
        dt = tf / (steps)

        # compile state trajectories
        xt = np.zeros((x0.shape[0], steps + 1))
        tt = np.zeros((steps + 1))
        Fit = np.zeros((steps + 1))
        Fot = np.zeros((steps + 1))

        # initialize simulation
        current_state = self.x0
        xt[:, 0] = current_state
        tt[0] = 0.
        Fit[0] = params["F_nom"]
        Fot[0] = 0.

        # simulation
        for s in range(steps):
            self.u = controls  # [:, s]                          # control for this step
            ode = scp.ode(model)  # define ode
            ode.set_integrator('lsoda', nsteps=3000)  # define integrator
            ode.set_initial_value(current_state, dt)  # set initial value
            current_state = list(ode.integrate(ode.t + dt))  # integrate system
            xt[:, s + 1] = current_state  # add current state
            tt[s + 1] = (s + 1) * dt
            Fit[s + 1] = self.F_in
            Fot[s + 1] = self.F_out
        reward = -abs(xt[0, -1] - 2)#  - 0.01* controls**2
        done = False

        return [(xt[0, -1]-2)/2, d], (reward-1)/1, done


class simple_CSTR:
    '''
    This files integrates the model.
    model: this is were the model should be changed
    '''

    # --- initializing model --- #
    def __init__(self):
        # Object variable definitions
        self.u_min = np.double(np.array([0.]))  # lower bound of inputs
        self.u_max = np.double(np.array([1.]))  # upper bound of inputs
        self.nx, self.nu, self.x_init = 1, 1, np.array([(0.2-0.5)/0.5])
        self.dt = 0.1
    # --- dynamic model definition --- #
    def model(self, t, state):
        # internal definitions
        u  = self.u
        x0 = 1.#self.x_init*0.5+0.5
        # define controls here
        uu = 3000*u[0]

        # state vector
        c = state[0]
        # differential equations
        dc = uu/5000 * (x0- c) - 2*c**3

        return np.array([dc], dtype='float64')

    # --- simulation --- #
    def reset(self):
        return self.x_init

    def simulation(self, controls, tf, x0):
        '''
        u shape -> (u_{dim},steps)
        '''

        # external definitions
        self.x0, self.tf = 0.5*x0+0.5, tf

        # internal definitions
        steps, model = 20, self.model

        dt = tf / (steps)

        # compile state trajectories
        xt = np.zeros((x0.shape[0], steps + 1))
        tt = np.zeros((steps + 1))


        # initialize simulation
        current_state = self.x0
        xt[:, 0] = current_state

        # simulation
        for s in range(steps):
            self.u = controls  # [:, s]                          # control for this step
            ode = scp.ode(model)  # define ode
            ode.set_integrator('lsoda', nsteps=3000)  # define integrator
            ode.set_initial_value(current_state, dt)  # set initial value
            current_state = list(ode.integrate(ode.t + dt))  # integrate system
            xt[:, s + 1] = current_state  # add current state
            tt[s + 1] = (s + 1) * dt

        reward = -(abs(xt[0, -1] - 0.3))#  - 0.01* controls**2
        done = False

        return [(xt[0, -1]-0.5)/0.5], (reward), done
