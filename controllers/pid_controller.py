# pid_controller.py

from utils.pid import get_pid_control, preprocess_obs

# -------------------------
# PID Configuration
# -------------------------
KP_X   = 1.25                                                                       # Proportional on cart position
KD_X   = 0.0                                                                        # Derivative on cart velocity
KP_TH  = 19.15                                                                      # Proportional on pole angle
KD_TH  = 6.43                                                                       # Derivative on pole angular velocity
KI_TH  = 0.22                                                                       # Optional integral on pole angle

# -------------------------
# Controller Function
# -------------------------
def pid_action(obs):
    """
    Computes the PPID control signal and returns discrete action for CartPole.

    Parameters:
        obs (list[float]): Environment state observation

    Returns:
        tuple: (int action, float control_signal, ndarray state_vector)
    """
    x = preprocess_obs(obs)
    u = get_pid_control(x, KP_X, KD_X, KP_TH, KD_TH, KI_TH)
    action = 0 if float(u) > 0 else 1
    return action, u, x
