import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0,1,0],
                      [0,0,1],
                      [0,0,0]])
        B = np.array([0,self.b,0])
        L = 50
        W = 2
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(np.array([0, self.b, 0]))


    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q, q_dot = x
        q_est = self.eso.get_state()
        q_dot_est = q_est[1]
        q_ddot_est = q_est[2]

        e = q - q_est[0]
        e_dot = q_dot - q_dot_est

        z1 = e - q_est[1]
        z2 = e_dot - q_dot_est

        z3 = q_dot - q_ddot_est

        u_tilde = self.kp * z1 + self.kd * z2 - q_ddot_est + self.b * z3

        u = u_tilde + q_d_ddot[0]

        self.eso.update(q, u)

        return u