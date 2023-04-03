import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)
        self.Kp = -0.5
        self.Kd = -0.5

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        print(x)
        v = q_r_ddot + self.Kd * (np.array([q1_dot, q2_dot]) - q_r_dot) + self.Kp * (np.array([q1, q2] - q_r))  # TODO: add feedback
        M = self.model.M(x)
        C = self.model.C(x)
        u = M @ v[:, np.newaxis] + C @ np.array([q1_dot, q2_dot])[:, np.newaxis]

        return u
