import pandas as pd
import matplotlib.pyplot as plt

files = {
    "PID": "results/csv_logs/run_pid_20250519_083521.csv",
    "LQR": "results/csv_logs/run_lqr_20250519_083800.csv",
    "CA": "results/csv_logs/run_ca_20250519_092701.csv",
    "DQN": "results/csv_logs/run_dqn_20250519_085419.csv"
}

moving_average_window = 5

avg_rewards = {}
for label, path in files.items():
    df = pd.read_csv(path)

    summed = df.groupby(["run_id", "episode_index"])["reward_received"].sum().reset_index()

    averaged = summed.groupby("episode_index")["reward_received"].mean()

    if moving_average_window:
        averaged = averaged.rolling(window=moving_average_window, min_periods=1).mean()

    avg_rewards[label] = averaged

avg_df = pd.DataFrame(avg_rewards)

plt.figure(figsize=(12, 6))
for label in avg_df.columns:
    plt.plot(avg_df.index, avg_df[label], label=label)

plt.xlabel("Episode")
plt.ylabel("Gjennomsnittlig totalbelønning")
plt.title("Sammenligning av kontrollere")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
