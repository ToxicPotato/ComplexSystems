# logger.py

import csv
import os
from datetime import datetime

# -------------------------
# Logger Setup
# -------------------------
# Provides functionality for creating a timestamped CSV logger file
# and recording each step taken by the CA controller during the episode.
# The logger saves each step to disk and prints summary output to console.


def create_logger(directory="logs", filename=None):
    """
    Creates a CSV logger file. If filename is provided, uses it.
    Otherwise, creates a timestamped file in the 'logs' directory.

    Parameters:
        directory (str): Path where log files will be saved (default is 'logs')
        filename (str): Optional filename (e.g., 'rule_005.csv')

    Returns:
        file handle: Opened file object for writing
        csv.DictWriter: Writer object to write dict rows to CSV
    """
    os.makedirs(directory, exist_ok=True)                                                           # Ensure log directory exists

    if filename is None:
        # Use timestamp as default filename if none is given
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}.csv"

    full_path = os.path.join(directory, filename)

    # Open the file for writing
    f = open(full_path, mode='w', newline='')

    # Define CSV columns including episode field
    writer = csv.DictWriter(f, fieldnames=[
        "episode", "step", "obs", "bit_pre", "bit_post", "action", "reward", "terminated"
    ])
    writer.writeheader()                                                                            # Write column headers

    return f, writer

# -------------------------
# Log a Single Step
# -------------------------
def log_step(writer, episode, step, obs, bit_pre, bit_post, action, reward, terminated):
    """
    Logs a single CA decision step into the CSV file.
    Also prints a summary to the terminal for real-time inspection.

    Parameters:
        writer (csv.DictWriter): CSV writer handle
        episode (int): Current episode number
        step (int): Timestep index in the episode
        obs (list): Raw float observations from the environment
        bit_pre (str): Bitstring before CA transformation
        bit_post (str): Bitstring after CA step(s)
        action (int): Chosen action (0 or 1)
        reward (float): Reward from environment after action
        terminated (bool): Whether the episode has ended
    """
    # Write one row of data to CSV
    writer.writerow({
        "episode": episode,
        "step": step,
        "obs": obs,
        "bit_pre": bit_pre,
        "bit_post": bit_post,
        "action": action,
        "reward": reward,
        "terminated": terminated
    })

    # Print a summary of the step to the console
    print(f"[Ep {episode:02} | Step {step:03}] Action: {action} | Reward: {reward} | Done: {terminated}")
    print(f"  Obs:     {obs}")
    print(f"  Bits in:  {bit_pre}")
    print(f"  Bits out: {bit_post}\n")