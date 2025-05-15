# pid.py

import numpy as np

# -------------------------
# State Variables (global)
# -------------------------
theta_integral = 0.0                                                                # Integral accumulator for pole angle error
KP_X = KD_X = KP_TH = KD_TH = KI_TH = 0.0                                           # PID gains, can be set externally

# -------------------------
# Gain Setter
# -------------------------
def set_pid_gains(kp_x, kd_x, kp_th, kd_th, ki_th):
    """
    Sets global PID gain parameters.

    Parameters:
        kp_x (float): Proportional gain for cart position
        kd_x (float): Derivative gain for cart velocity
        kp_th (float): Proportional gain for pole angle
        kd_th (float): Derivative gain for pole angular velocity
        ki_th (float): Integral gain for pole angle
    """
    global KP_X, KD_X, KP_TH, KD_TH, KI_TH
    KP_X = kp_x
    KD_X = kd_x
    KP_TH = kp_th
    KD_TH = kd_th
    KI_TH = ki_th


# -------------------------
# PID Control Logic
# -------------------------
def get_pid_control(x, KP_X, KD_X, KP_TH, KD_TH, KI_TH):
    """
    Computes PID control signal based on pole angle and angular velocity.

    Parameters:
        x (ndarray): 4x1 state column vector [x, x_dot, theta, theta_dot]
        KP_X (float): Proportional gain for cart position
        KD_X (float): Derivative gain for cart velocity
        KP_TH (float): Proportional gain for pole angle
        KD_TH (float): Derivative gain for pole angular velocity
        KI_TH (float): Integral gain for pole angle

    Returns:
        float: control signal u
    """
    global theta_integral

    # Extract state values
    pos = x[0, 0]                                                                   # cart position x
    vel = x[1, 0]                                                                   # cart velocity x_dot
    theta = x[2, 0]                                                                 # pole angle θ
    theta_dot = x[3, 0]                                                             # pole angular velocity θ̇

    # Update integral
    theta_integral += theta

    # Compute control signal using weighted combination of errors
    u = (
        - KP_X  * pos                                                               # Penalize deviation from center
        - KD_X  * vel                                                               # Dampen cart movement
        - KP_TH * theta                                                             # Correct pole angle
        - KD_TH * theta_dot                                                         # Dampen pole swing
        - KI_TH * theta_integral                                                    # Compensate for persistent angular error
    )

    return u


# -------------------------
# Observation Preprocessing
# -------------------------
def preprocess_obs(obs):
    """
    Converts raw CartPole observation to column vector.

    Parameters:
        obs (list[float] or ndarray): Environment state

    Returns:
        ndarray: 4x1 column vector
    """
    return np.array([[obs[0]], [obs[1]], [obs[2]], [obs[3]]])


# -------------------------
# Reset Integral Term
# -------------------------
def reset_pid():
    """
    Resets the internal integral accumulator for the PID controller.

    Should be called at the beginning of each new episode.
    """
    global theta_integral
    theta_integral = 0.0
