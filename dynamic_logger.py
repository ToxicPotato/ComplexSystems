import csv
import os
from datetime import datetime

def create_logger(directory="logs", filename=None, fieldnames=None):
    os.makedirs(directory, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}.csv"
    path = os.path.join(directory, filename)
    f = open(path, mode='w', newline='')
    if fieldnames is None:
        fieldnames = ["episode", "step", "obs", "action", "reward", "terminated"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer, fieldnames

def log_step(writer, fieldnames, **kwargs):
    # build a row dict with provided fields
    row = {key: kwargs.get(key, "") for key in fieldnames}
    writer.writerow(row)
    # basic printout
    epi   = kwargs.get('episode')
    stp   = kwargs.get('step')
    act   = kwargs.get('action')
    rew   = kwargs.get('reward')
    done  = kwargs.get('terminated')
    obs   = kwargs.get('obs')
    print(f"[Ep {epi:02} | Step {stp:03}] Action: {act} | Reward: {rew} | Done: {done}")
    print(f"  Obs: {obs}")
    # optional CA bits
    if 'bit_pre' in fieldnames:
        print(f"  Bits in:  {kwargs.get('bit_pre')}")
        print(f"  Bits out: {kwargs.get('bit_post')}")
    print()