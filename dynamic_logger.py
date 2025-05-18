# dynamic_logger.py
# A simple logger for recording experiment parameters and timestep data

import csv
import os
from datetime import datetime


# Directory where CSV logs will be saved
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


LOG_DIRECTORY = os.path.join(os.path.dirname(__file__), "results", "csv_logs")


def create_logger(controller_type, filename=None):
    """
    Creates a CSV logger for the given controller_type ('basic' or 'ca').
    Returns: (file_handle, csv_writer)
    """
    ensure_dir(LOG_DIRECTORY)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{controller_type}_{timestamp}.csv"

    full_path = os.path.join(LOG_DIRECTORY, filename)
    csv_file = open(full_path, mode='w', newline='')

    # Common fields for all controllers
    BASIC_LOG_FIELDS = [
        'run_id', 'episode_index', 'step_count',
        'time_start', 'time_end', 'time_delta_ms',
        'observation_state', 'action_taken',
        'reward_received', 'terminated'
    ]
    # Additional fields for CA controller
    CA_LOG_FIELDS = [
        'bits_per_value', 'row_length', 'neighborhood_radius',
        'num_ca_ticks', 'boundary_condition', 'action_decoding', 'rule_index'
    ]
    if controller_type == 'ca':
        fieldnames = BASIC_LOG_FIELDS + CA_LOG_FIELDS
    else:
        fieldnames = BASIC_LOG_FIELDS

    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    print(f"Logging to {full_path}")
    return csv_file, csv_writer


def log_step(
        csv_writer,
        run_id,
        controller_type,
        episode_index,
        step_count,
        time_start,
        time_end,
        time_delta_ms,
        observation_state,
        action_taken,
        reward_received,
        terminated,
        # CA parameters (only used if controller_type=='ca')
        bits_per_value=None,
        row_length=None,
        neighborhood_radius=None,
        num_ca_ticks=None,
        boundary_condition=None,
        action_decoding=None,
        rule_index=None
):
    """
    Logs one timestep. Pass CA params when controller_type=='ca'.
    """
    row = {
        'run_id': run_id,
        'episode_index': episode_index,
        'step_count': step_count,
        'time_start': time_start,
        'time_end': time_end,
        'time_delta_ms': time_delta_ms,
        'observation_state': observation_state,
        'action_taken': action_taken,
        'reward_received': reward_received,
        'terminated': terminated
    }
    if controller_type == 'ca':
        row.update({
            'bits_per_value': bits_per_value,
            'row_length': row_length,
            'neighborhood_radius': neighborhood_radius,
            'num_ca_ticks': num_ca_ticks,
            'boundary_condition': boundary_condition,
            'action_decoding': action_decoding,
            'rule_index': rule_index
        })
    csv_writer.writerow(row)
    # Console summary
    print(f"[Run {run_id} | Ep {episode_index:02} | Step {step_count:03}] "
          f"Action: {action_taken} | Reward: {reward_received} | Done: {terminated}")
