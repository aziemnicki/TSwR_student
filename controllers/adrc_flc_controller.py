import numpy
import numpy as np
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        p_1 = p[0]
        p_2 = p[1]
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.zeros((6,2))
        self.L = np.array([[3 * p_1, 0],
                           [0, 3 * p_2],
                           [3 * p_1 ** 2, 0],
                           [0, 3 * p_2 ** 2],
                           [p_1 ** 3, 0],
                           [0, p_2 ** 3]])
        self.W = np.zeros((2,6))
        self.W[:2, :2] = np.eye(2)
        self.A = np.zeros((6,6))
        self.A[2:4, 4:] = np.eye(2)
        self.A[:2, 2:4] = np.eye(2)
        self.B = np.zeros((6,2))
        self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate((q, q_dot), axis=0)
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)
        M_hat = - (M_inv @ C)
        self.A[2:4, 2:4] = M_hat
        self.B[2:4, :2] = M_inv
        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1,q2])
        e = q1 - q_d
        z_hat = self.eso.get_state()
        x_hat = z_hat[0:2]
        x_hatDot = z_hat[2:4]
        f = z_hat[4:]
        e_dot = x_hatDot - q_d_dot
        M_est = self.model.M(x)
        C_est = self.model.C(x)

        v = self.Kp@e + self.Kd@e_dot+q_d_dot
        u = M_est@(v-f) + C_est@x_hatDot
        self.update_params(x_hat, x_hatDot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u
