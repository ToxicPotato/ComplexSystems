import numpy as np

class CartPoleCAAgent:
    def __init__(self, thresholds=None):
        min_vals = [-2.4, -10.0, -0.2095, -10.0]
        max_vals = [2.4, 10.0, 0.2095, 10.0]

        if thresholds is None:
            self.thresholds = np.random.uniform(min_vals, max_vals)
        else:
            self.thresholds = np.array(thresholds)

    def act(self, observation):
        pos, vel, ang, vel_ang = observation
        pos_thresh, vel_thresh, ang_thresh, vel_ang_thresh = self.thresholds

        pos = np.clip(pos / 4.8, -1, 1)
        vel = np.clip(vel / 10.0, -1, 1)
        ang = np.clip(ang / 0.418, -1, 1)
        vel_ang = np.clip(vel_ang / 10.0, -1, 1)

        decision = (
            (pos / pos_thresh) +
            (vel / vel_thresh) +
            (ang / ang_thresh) +
            (vel_ang / vel_ang_thresh)
        )

        return 1 if decision > 0 else 0
    
    def mutate(self, mutation_rate=0.1):
        min_vals = [-2.4, -10.0, -0.2095, -10.0]
        max_vals = [2.4, 10.0, 0.2095, 10.0]

        self.thresholds += np.random.uniform(-mutation_rate, mutation_rate, size=4)
        self.thresholds = np.clip(self.thresholds, min_vals, max_vals)
    
    @staticmethod
    def crossover(parent1, parent2):
        new_thresholds = (parent1.thresholds + parent2.thresholds) / 2
        return CartPoleCAAgent(new_thresholds)
