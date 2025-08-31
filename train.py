# train.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import Agent

# Initialize the environment
env = gym.make("CartPole-v1")

# Create the agent
agent = Agent(env)

# Training parameters
num_episodes = 20000
episode_rewards = []
log_interval = 1000

print("Starting training...")

for episode in range(num_episodes):
    # Reset the environment for a new episode
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Agent chooses an action
        action = agent.choose_action(state)
        
        # Environment performs the action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Agent learns from the experience
        agent.learn(state, action, reward, next_state, done)
        
        # Update state and total reward
        state = next_state
        total_reward += reward

    # Decay epsilon after each episode
    agent.decay_epsilon()
    
    # Log rewards and progress
    episode_rewards.append(total_reward)
    if (episode + 1) % log_interval == 0:
        avg_reward = np.mean(episode_rewards[-log_interval:])
        print(f"Episode {episode + 1}/{num_episodes} | Average Reward (last {log_interval}): {avg_reward:.2f}")

env.close()
print("Training finished.")

#Save the trained Q-table
np.save('q_table.npy', agent.q_table)
print("Q-table saved to q_table.npy")

# --- Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')

# Calculate and plot a moving average of the rewards to see the trend
moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
plt.plot(moving_avg, label='Moving Average (100 episodes)', color='red', linewidth=3)

plt.title('CartPole Q-Learning Training Performance')
plt.xlabel('Episode')
plt.ylabel('Total Reward (Episode Duration)')
plt.legend()
plt.grid(True)
plt.savefig('training_performance.png')
print("Performance plot saved to training_performance.png")
plt.show()