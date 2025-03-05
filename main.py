import gymnasium as gym
import numpy as np
from model import CartPoleCAAgent

num_generations = 5
num_agents = 10
num_best_agents = 3
num_eval_episodes = 5

env = gym.make("CartPole-v1", render_mode="human")

population = [CartPoleCAAgent() for _ in range(num_agents)]

for generation in range(num_generations):
    print(f"Generasjon {generation + 1} starter...")

    agent_scores = np.zeros(num_agents)

    for i, agent in enumerate(population):
        print(f"\nEvaluering av agent {i+1}/{num_agents} - Terskler: {agent.thresholds}")

        for episode_num in range(num_eval_episodes):
            obs, info = env.reset()
            episode_over = False
            score = 0

            while not episode_over:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                # env.render()
                score += reward

                episode_over = terminated or truncated

            agent_scores[i] += score

    best_agents_idx = np.argsort(agent_scores)[-num_best_agents:][::-1]
    best_agents = [population[i] for i in best_agents_idx]

    print("\nBeste agenter i generasjonen:")
    for rank, idx in enumerate(best_agents_idx):
        print(f" - Agent {idx}: Score {agent_scores[idx]} | Terskler: {population[idx].thresholds}")

    new_population = []

    new_population.extend(best_agents)

    while len(new_population) < num_agents:
        parent1, parent2 = np.random.choice(best_agents, 2)
        child = CartPoleCAAgent.crossover(parent1, parent2)
        child.mutate()
        new_population.append(child)

    population = new_population

env.close()
print(f"Evolusjon ferdig etter {num_generations} generasjoner!")
