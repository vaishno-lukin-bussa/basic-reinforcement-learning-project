# play.py

import gymnasium as gym
import numpy as np
import time
import pygame
from q_learning_agent import Agent

# Initialize the environment with rendering
env = gym.make("CartPole-v1", render_mode='human')

# Create an agent instance (we only need it for its state discretization logic)
agent = Agent(env)

# Load the trained Q-table
try:
    agent.q_table = np.load('q_table.npy')
    print("Q-table loaded successfully.")
except FileNotFoundError:
    print("Error: 'q_table.npy' not found. Please run train.py first.")
    exit()

# Set epsilon to 0 to ensure the agent only uses the learned policy (no exploration)
agent.epsilon = 0

# Initialize pygame font system
pygame.init()
font = pygame.font.SysFont("Arial", 24)

# --- Watch the agent play one episode ---
state, _ = env.reset()
done = False
total_reward = 0
step_count = 0

print("\nStarting playback of the trained agent...")

while not done:
    # Render the environment
    env.render()
    
    # Choose the best action from the Q-table
    action = agent.choose_action(state)
    
    # Perform the action
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    state = next_state
    total_reward += reward
    step_count += 1

    # --- Draw steps on the pygame window ---
    surface = pygame.display.get_surface()
    if surface:  # Only if a window exists
        # Render step text
        text = font.render(f"Steps: {step_count}", True, (255, 0, 0))
        surface.blit(text, (10, 10))  # Draw text at top-left
        pygame.display.flip()

    # Small delay to make playback watchable
    time.sleep(0.02)

print(f"Playback finished. Total reward: {total_reward}, Steps survived: {step_count}")
env.close()
pygame.quit()
