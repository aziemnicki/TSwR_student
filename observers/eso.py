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
        self.states.append(self.state)
        ### TODO implement ESO update
        z = np.reshape(self.state, (len(self.state), 1))

        zhatdot = np.dot(self.A, z) + np.dot(self.B, u) + np.dot(self.L, (q - self.W @ z))
        self.state = (self.state + self.Tp * np.reshape(zhatdot, (1, len(zhatdot))))[0]

        return self.state


    def get_state(self):
            return self.state
