import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        model_1 = ManipulatorModel(Tp)
        model_1.r3 = 0.05
        model_1.m3 = 0.1
        model_2 = ManipulatorModel(Tp)
        model_2.r3 = 0.01
        model_2.m3 = 0.01
        model_3 = ManipulatorModel(Tp)
        model_3.r3 = 0.3
        model_3.m3 = 1.0
        self.models = [model_1, model_2, model_3]
        self.i = 0
        self.u_prev = np.array([[0], [0]])
        self.x_prev = np.array([0, 0, 0, 0])
        self.x_compare = [0, 0, 0]
        self.Tp = Tp
        self.x_n = np.zeros((3, 4))
        self.x_ndot_prev = np.zeros((3, 4))
        self.x_ndot_prev2 = np.zeros((3, 4))
        #feedback
        self.Kp = -1
        self.Kd = -1

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        err = np.zeros(3)
        for i, model in enumerate(self.models):
            x_ndot = model.x_dot(self.x_prev, self.u_prev)
            self.x_ndot_prev[i][0] = self.x_ndot_prev[i][0] + ((self.x_ndot_prev2[i][0] + x_ndot[0]) / 2 * self.Tp)
            #print(self.x_n[0])
            self.x_ndot_prev[i][1] = self.x_ndot_prev[i][1] + ((self.x_ndot_prev2[i][1] + x_ndot[1]) / 2 * self.Tp)
            # self.x_ndot_prev[i][2] = x_ndot[0]
            # self.x_ndot_prev[i][3] = x_ndot[1]
            self.x_ndot_prev[i][2] = self.x_prev[2]+((x_ndot[2]+self.x_ndot_prev[i][2])/2*self.Tp)
            self.x_ndot_prev[i][3] = self.x_prev[3]+((x_ndot[3]+self.x_ndot_prev[i][3])/2*self.Tp)
            self.x_ndot_prev2[i] = x_ndot.ravel()
            err[i] = (abs(x[0] - self.x_ndot_prev2[i][0]) + abs(x[2]-self.x_ndot_prev2[i][2])) / 2
        print('model nr')
        print(err)
        self.i = np.argmin(err)
        self.x_ndot_prev2 = self.x_ndot_prev
        self.x_prev = x

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q1, q2, q1_dot, q2_dot = x
        v = q_r_ddot + self.Kd * (np.array([q1_dot, q2_dot]) - q_r_dot) + self.Kp * (np.array([q1, q2] - q_r))
        #v= q_r_ddot
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ np.array([q1_dot, q2_dot])[:, np.newaxis]
        self.u_prev = u
        return u
