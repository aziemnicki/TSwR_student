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
        self.eso.B[1] = self.b


    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC

        # Calculate the error and its derivatives
        e = q_d - x
        e_dot = q_d_dot - np.dot(self.eso.W, self.eso.state)
        A_state = np.dot(self.eso.A, self.eso.state) if self.eso.state is not None else 10
        B_kp_e = np.dot(self.eso.B, self.kp * e) if self.kp is not None else 10
        L_e_dot = np.dot(self.eso.L, e_dot) if e_dot is not None else 10
        e_ddot = q_d_ddot - A_state - B_kp_e - L_e_dot

        # Update the ESO with the new state estimate
        self.eso.update(q_d_ddot, self.kd * e_dot + self.kp * e) if self.kd is not None else 0

        # Calculate the control input using ADRC equations
        v = e_ddot + np.dot(self.eso.L, e_dot) + np.dot(self.eso.W, self.eso.state)
        u = (1 / self.b) * (v - np.dot(self.eso.A, self.eso.state) - np.dot(self.eso.B, self.kd * e_dot) - np.dot(
            self.eso.L, e)) if self.b is not None else 0


        return u