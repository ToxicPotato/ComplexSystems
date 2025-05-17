# lqr_controller.py
from utils.lqr.lqr import get_system_matrices, get_lqr_gain, preprocess_obs

# -------------------------
# System Configuration
# -------------------------
MASS_CART  = 1.0          # kg                                                      # Mass of the cart
MASS_POLE  = 0.1          # kg                                                      # Mass of the pole
POLE_LENGTH = 0.5         # m                                                       # Distance to the pole's center of mass

# LQR tuning (state and control cost)
Q = [1.0, 1.0, 10.0, 1.0]                                                           # Q matrix weights for [x, x_dot, theta, theta_dot]
R = 0.001                                                                           # R value penalizes excessive control force

# -------------------------
# Controller Initialization
# -------------------------

# Generate A and B matrices using current system configuration
A, B = get_system_matrices(m=MASS_POLE, M=MASS_CART, l=POLE_LENGTH)

# Calculate optimal LQR gain matrix K based on system and cost weights
K = get_lqr_gain(A, B, Q_vals=Q, R_val=R)

# -------------------------
# Controller Function
# -------------------------
def lqr_action(obs):
    """
    Computes the LQR control signal and returns discrete action for CartPole.

    Parameters:
        obs (list[float]): Environment state observation

    Returns:
        tuple: (int action, float control_signal, ndarray state_vector)
    """
    x = preprocess_obs(obs)                                                         # Convert observation to 4x1 column vector
    u = -K @ x                                                                      # Compute continuous control signal u = -Kx
    action = 1 if float(u) > 0 else 0                                               # Convert control to discrete action: 1 (right) or 0 (left)
    return action, u, x                                                             # Return action, control signal, and processed state

