import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0], [self.b], [0]])
        L = np.array([[3*p],
                      [3*p**2],
                      [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0], [self.b], [0]])
        self.eso.set_B(B)


    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        ### TODO 2 implement ADRC

        # # Calculate the error and its derivatives
        e = x[0] - q_d
        z_hat = self.eso.get_state()
        x_hat = z_hat[0]
        x_hatDot = z_hat[1]
        f = z_hat[2]
        e_dot = x_hatDot - q_d_dot
        v = (q_d_ddot + self.kd * e_dot + self.kp * e) - f
        u = (1/self.b) * v if self.b is not None else 1

        self.eso.update(x[0], u)
        #TODO 9

        # model M matrix
        l1 = 0.5
        r1 = 0.04
        m1 = 3.0
        l2 = 0.4
        r2 = 0.04
        m2 = 2.4
        I_1 = 1 / 12 * m1 * (3 * r1 ** 2 + l1 ** 2)
        I_2 = 1 / 12 * m2 * (3 * r2 ** 2 + l2 ** 2)
        m3 = 0.8
        r3 = 0.05
        I_3 = 2. / 5 * m3 * r3 ** 2
        d1 = l1 / 2
        d2 = l2 / 2
        if i % 2 == 0:
            alpha = I_1 + I_2 + m1 * d1 ** 2 + m2 * (l1 ** 2 + d2 ** 2) + I_3 + m3 * (l1 ** 2 + l2 ** 2)
            beta = m2 * l1 * d2 + m3 * l1 * l2
            delta = I_2 + m2 * d2 ** 2 + I_3 + m3 * l2 ** 2

            m_00 = alpha + 2 * beta * np.cos(x_hat)
            m_01 = delta + beta * np.cos(x_hat)
            m_10 = m_01
            m_11 = delta
            # inversing matrix M to M^-1
            M = [[m_00, m_01], [m_10, m_11]]
            M_inv = np.linalg.inv(M)
            self.set_b(M_inv[i, i])
        else: pass

        return u