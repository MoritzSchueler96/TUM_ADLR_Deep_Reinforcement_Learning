import numpy as np


class PIDController:
    def __init__(self, dt=0.01):
        #used

        # init one PI controllers for velocity and two PID controller for phi and theta
        self.k_p_V = 0.5
        self.k_i_V = 0.1
        self.k_p_phi = 1
        self.k_i_phi = 0  # Note gain is set to zero for roll channel
        self.k_d_phi = 0.5
        self.k_p_theta = -4
        self.k_i_theta = -0.75
        self.k_d_theta = -0.1

        # set physical limits
        self.delta_a_min = np.radians(-30)
        self.delta_e_min = np.radians(-30)
        self.delta_a_max = np.radians(30)
        self.delta_e_max = np.radians(35)

        self.dt = dt

        # init velocity, pitch and roll angles
        self.va_r = None
        self.phi_r = None
        self.theta_r = None

        # init integral part of velocity, pitch and roll angles
        self.int_va = 0
        self.int_roll = 0
        self.int_pitch = 0

    def set_reference(self, phi, theta, va):
        """
        Set velocity, pitch and roll angle manually.
        :param phi: pitch angle.
        :param theta: roll angle.
        :param va: velocity.
        :return:
        """
        #used
        self.va_r = va
        self.phi_r = phi
        self.theta_r = theta

    def reset(self):
        """
        Reset velocity, pitch and roll angle manually.
        :return:
        """
        self.int_va = 0
        self.int_roll = 0
        self.int_pitch = 0

    def get_action(self, phi, theta, va, omega):
        """
        Ca.
        :param phi: pitch angle.
        :param theta: roll angle.
        :param va: velocity.
        :param omega: #TODO: describe omega
        :return:
        """
        #used
        # calculate control deviation (P)
        e_V_a = va - self.va_r
        e_phi = phi - self.phi_r
        e_theta = theta - self.theta_r

        # (integral states are initialized to zero)
        # calculate control deviation (I)
        self.int_va = self.int_va + self.dt * e_V_a
        self.int_roll = self.int_roll + self.dt * e_phi
        self.int_pitch = self.int_pitch + self.dt * e_theta

        # Note the different sign on pitch gains below.
        # Positive aileron  -> positive roll moment
        # Positive elevator -> NEGATIVE pitch moment

        # Update actual state
        delta_t = 0 - self.k_p_V * e_V_a - self.k_i_V * self.int_va  # PI

        delta_a = (
            -self.k_p_phi * e_phi
            - self.k_i_phi * self.int_roll
            - self.k_d_phi * omega[0]
        )  # PID

        delta_e = (
            0
            - self.k_p_theta * e_theta
            - self.k_i_theta * self.int_pitch
            - self.k_d_theta * omega[1]
        )  # PID

        delta_r = 0  # No rudder available # TODO: delete as not needed?

        # Constrain input
        delta_t = np.clip(delta_t, 0, 1.0)  # throttle between 0 and 1
        # TODO: maybe use sth different than clip

        delta_a = np.clip(delta_a, self.delta_a_min, self.delta_a_max)
        delta_e = np.clip(delta_e, self.delta_e_min, self.delta_e_max)

        return np.asarray([delta_e, delta_a, delta_t])

