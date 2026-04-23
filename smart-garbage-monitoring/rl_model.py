import numpy as np
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/garbage_large_dataset.csv")

# Convert fill level to discrete states
def get_state(fill):
    if fill < 30:
        return 0   # Empty
    elif fill < 75:
        return 1   # Half
    else:
        return 2   # Full

df["State"] = df["Fill_Level"].apply(get_state)

# Actions: 0=Wait, 1=Schedule, 2=Priority
actions = [0, 1, 2]

# Initialize Q-table (3 states x 3 actions)
Q = np.zeros((3, 3))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 5000

for episode in range(episodes):
    sample_data = df.sample(100)

    for index, row in sample_data.iterrows():
        state = row["State"]

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q[state])

        # Reward logic
        if state == 2:  # Full
            if action == 2:
                reward = 15
            elif action == 1:
                reward = 5
            else:
                reward = -25

        elif state == 1:  # Half
            if action == 1:
                reward = 8
            elif action == 2:
                reward = 2
            else:
                reward = -5

        else:  # Empty
            if action == 0:
                reward = 5
            else:
                reward = -10

        next_state = state

        # Q-learning update formula
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

print("Final Q-Table:")
print(Q)

# Save Q-table
joblib.dump(Q, "models/rl_q_table.pkl")

print("Reinforcement Learning model trained and saved!")
