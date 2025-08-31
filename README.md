# 🏋️‍♂️ Q-Learning Agent for CartPole

This project implements a **Q-Learning agent** to solve the classic **CartPole-v1** environment from [Gymnasium](https://gymnasium.farama.org/).  
The agent learns to balance a pole on a moving cart by interacting with the environment and updating a **Q-table**.

---

## 📂 Project Structure

```
├── play.py                   #Run a trained agent and watch it play
├── q_learning_agent.py       #Q-learning Agent implementation
├── q_table.npy               #Saved Q-table after training
├── requirements.txt          #Dependencies
├── train.py                  #Script to train the agent
├── training_performance.png  #Reward curve over training
```

---

## ⚙️ How It Works

### 1. Environment
The agent is trained in the **CartPole-v1** environment, where:
- The cart can move **left** or **right**.
- The goal is to prevent the pole from falling by applying the right forces.

### 2. State Discretization
Since CartPole has **continuous states**, we discretize them into buckets:
- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity  

This converts the state space into a finite grid for a **Q-table** lookup.

### 3. Q-Learning Algorithm
The agent updates its **Q-values** using the Bellman equation:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \big]
\]

- **s**: current state  
- **a**: action taken  
- **r**: reward received  
- **s'**: next state  
- **α (alpha)**: learning rate  
- **γ (gamma)**: discount factor  

### 4. Exploration vs Exploitation
The agent uses an **epsilon-greedy policy**:
- With probability **ε**, it explores (random action).  
- With probability **1-ε**, it exploits the best action from the Q-table.  
- ε decays over time → more exploitation as training progresses.

---

## 🎯 Reward Calculation

In **CartPole-v1**, the reward is provided **by the environment**:
- **+1 reward for every time step the pole remains upright**  
- If the pole falls or the cart goes out of bounds → **episode ends**

So, the **total reward for an episode** equals the **duration (steps survived)**.

For example:
- If the pole stays balanced for 250 steps → **Reward = 250**  
- If it falls at 50 steps → **Reward = 50**

The Q-learning agent uses this feedback to improve its policy.

---

## 🚀 Training the Agent

Run:
```bash
python train.py
```

This will:
- Train the agent for **20,000 episodes**
- Save the learned **q_table.npy**
- Generate a **performance plot (training_performance.png)**

---

## 🎮 Watching the Agent Play

Once trained, watch the agent balance the pole:
```bash
python play.py
```

The agent will:
- Load the trained **q_table.npy**
- Play one episode using the learned policy (no random actions)
- Display **Steps survived** directly in the Pygame window

---

## 📊 Results

Training performance is logged and visualized:  
- **Blue line** → reward per episode  
- **Red line** → moving average (trend of learning progress)  

Example plot (`training_performance.png`):  
![Training Performance](training_performance.png)

---

## 📦 Requirements


### Create and Activate Virtual Environment (Windows PowerShell)

Before installing dependencies, create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `gymnasium[classic-control]`
- `numpy`
- `matplotlib`
- `pygame`

---

## 📝 Summary

- Implements **Q-learning** from scratch  
- Solves **CartPole-v1** using **state discretization**  
- Stores learned policy in **Q-table**  
- Visualizes training performance  
- Displays **steps survived in real-time** on the Pygame window  

---
