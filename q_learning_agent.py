# q_learning_agent.py

import numpy as np

class Agent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space_size = env.action_space.n

        # Define the size of the buckets for discretizing the state space
        # Format: [cart_position, cart_velocity, pole_angle, pole_velocity]
        self.num_buckets = (10, 10, 12, 12) 

        # Define the boundaries for each state variable
        # Note: Cart velocity and pole velocity can be high, so we clip them to a reasonable range.
        self.state_bounds = [
            (-4.8, 4.8),
            (-4.0, 4.0),  # Clamped range for cart velocity
            (-0.418, 0.418), # ~24 degrees in radians
            (-4.0, 4.0)   # Clamped range for pole velocity
        ]

        # Initialize the Q-table with zeros
        q_table_shape = self.num_buckets + (self.action_space_size,)
        self.q_table = np.zeros(q_table_shape)

    def discretize_state(self, state):
        """Converts a continuous state into a discrete one using buckets."""
        discrete_state = []
        for i in range(len(state)):
            # If state value is outside the bounds, clip it
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.num_buckets[i] - 1
            else:
                # Calculate the bucket index for the current state variable
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (state[i] - self.state_bounds[i][0]) / bound_width
                scaling = offset * self.num_buckets[i]
                bucket_index = int(scaling)
            discrete_state.append(bucket_index)
        return tuple(discrete_state)

    def choose_action(self, state):
        """Chooses an action using the epsilon-greedy policy."""
        # Exploration vs. Exploitation
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the best action from the Q-table
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, done):
        """Updates the Q-table using the Bellman equation."""
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)

        # If the episode is over, the future Q-value is 0
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_discrete_state])

        current_q = self.q_table[discrete_state + (action,)]

        # Q-learning formula
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        
        # Update the Q-table
        self.q_table[discrete_state + (action,)] = new_q

    def decay_epsilon(self):
        """Reduces epsilon to decrease exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)