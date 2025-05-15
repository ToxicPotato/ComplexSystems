import csv
import os
from datetime import datetime

BASE_DIR    = os.path.dirname(__file__)
DEFAULT_DIR = os.path.join(BASE_DIR, "results", "csv_logs")

def create_logger(directory=DEFAULT_DIR, filename=None, fieldnames=None):
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"run_{timestamp}.csv"

    full_path   = os.path.join(directory, filename)
    file_handle = open(full_path, mode='w', newline='')

    if fieldnames is None:
        fieldnames = [
            "episode_index",
            "step_count",
            "observation_state",
            "action_taken",
            "reward_received",
            "terminated"
        ]

    csv_writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    csv_writer.writeheader()

    return file_handle, csv_writer, fieldnames

def log_step(
    csv_writer,
    fieldnames,
    episode_index,
    step_count,
    observation_state,
    action_taken,
    reward_received,
    terminated,
    bit_pre="",
    bit_post=""
):
    # Build the row
    row = {}
    for column in fieldnames:
        if column == "episode_index":
            row[column] = episode_index
        elif column == "step_count":
            row[column] = step_count
        elif column == "observation_state":
            row[column] = observation_state
        elif column == "bit_pre":
            row[column] = bit_pre
        elif column == "bit_post":
            row[column] = bit_post
        elif column == "action_taken":
            row[column] = action_taken
        elif column == "reward_received":
            row[column] = reward_received
        elif column == "terminated":
            row[column] = terminated
        else:
            row[column] = ""

    # Write to CSV
    csv_writer.writerow(row)

    # Print summary
    print(f"[Ep {episode_index:02} | Step {step_count:03}] "
          f"Action: {action_taken} | Reward: {reward_received} | Done: {terminated}")
    print(f"  Obs: {observation_state}")
    if "bit_pre" in fieldnames:
        print(f"  Bits in:  {bit_pre}")
        print(f"  Bits out: {bit_post}")
    print()
