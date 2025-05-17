# pid_optimize.py

import numpy as np
import gymnasium as gym
import csv
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing import freeze_support

# -------------------------
# Search Configuration
# -------------------------
# Define the ranges of PID parameters to search over
KP_X_VALUES = np.linspace(0.0, 5.0, 5)                                              # Proportional gain for cart position (x)
KD_X_VALUES = np.linspace(0.0, 5.0, 5)                                              # Derivative gain for cart velocity (x_dot)
KP_TH_VALUES = np.linspace(1.0, 60.0, 14)                                           # Proportional gain for pole angle (theta)
KD_TH_VALUES = np.linspace(0.0, 15.0, 15)                                           # Derivative gain for pole angular velocity (theta_dot)
KI_TH_VALUES = np.linspace(0.0, 2.0, 10)                                            # Integral gain for pole angle (theta)

NUM_EPISODES_PER_COMBO = 3                                                          # How many episodes to run per PID combination
MAX_STEPS = 100000                                                                  # Max steps per episode (custom override)
LOG_FILE = "results/pid_grid_log.csv"                                               # File to log results from all combinations

# -------------------------
# Evaluation Function (runs in subprocess)
# -------------------------
def evaluate_pid_combination(args):
    """
    Evaluates a given PID parameter combination on the CartPole-v1 environment.

    Parameters:
        args (tuple): A 5-tuple of PID values: (kp_x, kd_x, kp_th, kd_th, ki_th)

    Returns:
        tuple: (avg_score, args) where avg_score is the average steps balanced
                over NUM_EPISODES_PER_COMBO episodes.

    Note:
        Uses a basic PID formula: u = -(Kp*x + Kd*x_dot + Kp_theta*theta + Kd_theta*theta_dot + Ki_theta*integral_theta)
        Converts continuous force u into discrete action (0 or 1).
    """
    kp_x, kd_x, kp_th, kd_th, ki_th = args
    total_steps = []

    for _ in range(NUM_EPISODES_PER_COMBO):
        env = gym.make("CartPole-v1")                                               # Create a fresh environment
        obs, _ = env.reset()
        steps = 0
        theta_integral = 0.0                                                        # Reset integral term

        for _ in range(MAX_STEPS):
            # Convert observation to column vector
            x = np.array([[obs[0]], [obs[1]], [obs[2]], [obs[3]]])
            theta = x[2, 0]
            theta_dot = x[3, 0]
            pos = x[0, 0]
            vel = x[1, 0]

            # Accumulate integral of the pole angle over time
            theta_integral += theta

            # PID control law for balancing the pole
            u = (
                - kp_x * pos
                - kd_x * vel
                - kp_th * theta
                - kd_th * theta_dot
                - ki_th * theta_integral
            )

            # Convert control output to discrete action (left or right force)
            action = 0 if float(u) > 0 else 1

            # Step the environment
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated:
                break

        total_steps.append(steps)
        env.close()

    avg_score = np.mean(total_steps)                                                # Average number of steps survived
    return avg_score, args


# -------------------------
# CSV Logger
# -------------------------
def log_result_csv(filepath, header, row):
    """
    Logs a dictionary of values as a row to a CSV file.

    Parameters:
        filepath (str): Path to the output CSV file
        header (list): Column names for the CSV
        row (dict): Dictionary of values to write as a row
    """
    exists = os.path.isfile(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------
# Parallel Grid Search Optimizer
# -------------------------
def optimize_pid_gains():
    """
    Performs a parallel grid search across all PID combinations.

    Submits each combination to a process pool and collects the average score.
    Logs each result to CSV and prints the best combination at the end.

    Returns:
        tuple: The best-performing PID parameter combination.
    """
    param_combos = list(product(
        KP_X_VALUES,
        KD_X_VALUES,
        KP_TH_VALUES,
        KD_TH_VALUES,
        KI_TH_VALUES
    ))

    best_combo = None
    best_score = 0.0

    print(f"Evaluating {len(param_combos)} combinations in parallel...\n")

    header = ['kp_x', 'kd_x', 'kp_th', 'kd_th', 'ki_th', 'avg_steps']
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)                                                         # Clear previous results

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_pid_combination, combo) for combo in param_combos]

        for future in as_completed(futures):
            score, args = future.result()
            kp_x, kd_x, kp_th, kd_th, ki_th = args

            print(f"Tested KP_TH={kp_th:.2f}, KD_TH={kd_th:.2f}, KI_TH={ki_th:.2f}, "
                  f"KP_X={kp_x:.2f}, KD_X={kd_x:.2f} → {score:.1f} steps")

            log_result_csv(LOG_FILE, header, {
                'kp_x': kp_x,
                'kd_x': kd_x,
                'kp_th': kp_th,
                'kd_th': kd_th,
                'ki_th': ki_th,
                'avg_steps': score
            })

            if score > best_score:
                best_score = score
                best_combo = args

    # Print best configuration
    kp_x, kd_x, kp_th, kd_th, ki_th = best_combo
    print("\nBest PID combo:")
    print(f"  KP_X={kp_x:.2f}")
    print(f"  KD_X={kd_x:.2f}")
    print(f"  KP_TH={kp_th:.2f}")
    print(f"  KD_TH={kd_th:.2f}")
    print(f"  KI_TH={ki_th:.2f}")
    print(f"  → Avg: {best_score:.1f} steps")

    return best_combo


# -------------------------
# Plot Results
# -------------------------
def plot_pid_results(filepath=LOG_FILE):
    """
    Loads the logged results from CSV and generates a heatmap
    showing how average steps vary with KP_TH and KD_TH.

    Parameters:
        filepath (str): Path to the CSV file containing grid search results
    """
    df = pd.read_csv(filepath)

    # Pivot data for heatmap
    pivot = df.groupby(['kp_th', 'kd_th'])['avg_steps'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title("PID Performance Heatmap (avg steps)")
    plt.xlabel("KD_TH")
    plt.ylabel("KP_TH")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/pid_heatmap.png")
    plt.show()

# -------------------------
# Main Entry Point
# -------------------------
if __name__ == "__main__":
    freeze_support()                                                                # Required for Windows multiprocessing
    best_combo = optimize_pid_gains()                                               # Run the grid search
    plot_pid_results()                                                              # Visualize the results