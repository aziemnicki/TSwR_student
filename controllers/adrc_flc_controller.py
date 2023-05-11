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

        self.eso.A = None
        self.eso.B = None

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        e = x[0] - q_d
        z_hat = self.eso.get_state()
        x_hat = z_hat[0]
        x_hatDot = z_hat[1]
        f = z_hat[2]
        e_dot = x_hatDot - q_d_dot
        v = (q_d_ddot + self.kd * e_dot + self.kp * e) - f
        u = (1 / self.b) * v if self.b is not None else 1

        self.eso.update(x[0], u)
        return u
