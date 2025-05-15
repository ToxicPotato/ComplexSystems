# lqr.py

import numpy as np
from scipy.linalg import solve_continuous_are

# -------------------------
# Static Constants
# -------------------------
GRAVITY = 9.8  # m/s^2

# -------------------------
# System Matrices Generator
# -------------------------
def get_system_matrices(m, M, l, g=GRAVITY):
    """
    Generates the linearized A and B matrices for the inverted pendulum system.

    Parameters:
        m (float): Mass of the pole
        M (float): Mass of the cart
        l (float): Length to center of mass of the pole
        g (float): Gravitational acceleration (default = GRAVITY)

    Returns:
        A (ndarray): State matrix (4x4)
        B (ndarray): Control input matrix (4x1)
    """

    # State matrix A represents the dynamics of the state vector
    A = np.array([
        [0, 1, 0, 0],                                                               # x_dot = x1
        [0, 0, -(m * g) / M, 0],                                                    # x1_dot depends on pole angle
        [0, 0, 0, 1],                                                               # theta_dot = x3
        [0, 0, ((M + m) * g) / (l * M), 0]                                          # theta_dot_dot from force and gravity
    ])

    # Input matrix B defines how control input (force) affects the system
    B = np.array([
        [0],                                                                        # Force has no effect on position directly
        [1 / M],                                                                    # Force affects cart acceleration
        [0],                                                                        # No direct effect on pole angle
        [-1 / (l * M)]                                                              # Force indirectly affects pole angle acceleration
    ])

    return A, B

# -------------------------
# LQR Gain Computation
# -------------------------
def get_lqr_gain(A, B, Q_vals, R_val):
    """
    Solves the continuous-time Algebraic Riccati Equation (ARE) and returns LQR gain.

    Parameters:
        A (ndarray): State matrix
        B (ndarray): Input matrix
        Q_vals (list[float]): Diagonal values for Q matrix
        R_val (float): Scalar for R matrix

    Returns:
        K (ndarray): Gain matrix such that u = -Kx
    """
    Q = np.diag(Q_vals)                                                             # Create diagonal Q matrix from weights
    R = np.array([[R_val]])                                                         # Create scalar R matrix
    P = solve_continuous_are(A, B, Q, R)                                            # Solve the Riccati equation for cost-to-go matrix P
    K = np.linalg.inv(R) @ B.T @ P                                                  # Compute optimal gain matrix: K = R⁻¹ * Bᵗ * P
    return K

# -------------------------
# Observation Preprocessing
# -------------------------
def preprocess_obs(obs):
    """
    Converts observation vector into column vector for matrix math.

    Parameters:
        obs (list[float] or ndarray): [x, x_dot, theta, theta_dot]

    Returns:
        ndarray: 4x1 column vector
    """
    return np.array([[obs[0]], [obs[1]], [obs[2]], [obs[3]]])
