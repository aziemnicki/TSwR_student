from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []
        print(self.state)
    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update

        y = np.dot(self.W, self.state)  # output
        e = q - y
        z = np.dot(self.B, u) - self.L * e
        xdot = np.dot(self.A, self.state) + z
        dhatdot = xdot[-1] - self.L * e[-1]
        xhatdot = np.dot(self.A, self.state) + z - np.dot(self.W, np.array([[0], [dhatdot]]))

        # Update the ESO state
        self.state[:-1] = self.state[1:]
        self.state[-1] = xhatdot[-1] * self.Tp + self.state[-1]

        return self.state


    def get_state(self):
            return self.state
