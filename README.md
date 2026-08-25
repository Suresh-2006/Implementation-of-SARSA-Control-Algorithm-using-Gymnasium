# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that enables the agent to select better actions for reaching the goal state while avoiding the hole.

---

## Problem Statement

Implement the **SARSA (State-Action-Reward-State-Action) control algorithm** using the Gymnasium `FrozenLake-v1` environment. The agent must learn the optimal action-value function through interaction with the environment and gradually improve its policy for reaching the goal.

The implementation uses an **epsilon-greedy policy with variable epsilon**, where the exploration rate decreases gradually during training. The starting state and terminal states are determined by the Gymnasium environment rather than being manually assigned fixed state numbers.

---

## Software Requirements

* Python 3.x
* Gymnasium
* NumPy
* Matplotlib

### Installation

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment with a custom 4 × 4 map.

```text
FFSF
FFFF
FFFF
FFGF
```

The symbols represent:

| Symbol | Meaning             |
| ------ | ------------------- |
| `F`    | Frozen/safe surface |
| `S`    | Starting state      |
| `G`    | Goal state          |
| `H`    | Hole                |

The environment is created using:

```python
env = gym.make(
    "FrozenLake-v1",
    desc=[
        "FFSF",
        "FFFF",
        "FFFF",
        "FFGF"
    ],
    is_slippery=True
)
```

Since `is_slippery=True`, the agent's movement is stochastic. The actual next state may differ from the intended movement, making the learning problem more challenging.

The environment itself determines the initial state using:

```python
state, info = env.reset()
```

The program does not manually assign a fixed initial state such as `state = 0`. Similarly, terminal states are identified using the `terminated` and `truncated` signals returned by the environment.

The environment contains **16 states** and **4 possible actions**.

The actions are:

| Action | Direction |
| ------ | --------- |
| `0`    | Left      |
| `1`    | Down      |
| `2`    | Right     |
| `3`    | Up        |

---

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It is an **on-policy temporal-difference control algorithm**. SARSA updates the Q-value using the action that is actually selected by the current policy in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol    | Meaning                                       |
| --------- | --------------------------------------------- |
| $S_t$     | Current state                                 |
| $A_t$     | Current action                                |
| $R_{t+1}$ | Reward received after taking action $A_t$     |
| $S_{t+1}$ | Next state                                    |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$  | Learning rate                                 |
| $\gamma$  | Discount factor                               |
| $Q(s,a)$  | Action-value function                         |

In this implementation:

```text
Learning rate (α) = 0.1
Discount factor (γ) = 0.99
```

The Q-table is initialized with zeros and is updated after every interaction with the environment.

---

## Epsilon-Greedy Policy

SARSA uses an **epsilon-greedy policy** for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

The implementation uses a **variable epsilon** rather than keeping epsilon constant.

The initial epsilon is:

```text
ε = 1.0
```

The epsilon value is gradually reduced after each episode using:

```python
epsilon = max(
    epsilon_min,
    epsilon * epsilon_decay
)
```

The parameters used are:

```text
Initial epsilon = 1.0
Minimum epsilon = 0.05
Epsilon decay = 0.9995
```

This allows the agent to explore more during the beginning of training and gradually exploit the learned Q-values as training progresses.

---

## Algorithm

### SARSA Control Algorithm

1. Create the FrozenLake environment.
2. Obtain the number of states and actions from the environment.
3. Initialize the Q-table with zeros.
4. Set the learning rate, discount factor, and epsilon parameters.
5. Reset the environment to obtain the initial state.
6. Select the first action using the epsilon-greedy policy.
7. Take the selected action in the environment.
8. Observe the next state, reward, and terminal status.
9. If the episode has not terminated, select the next action using the epsilon-greedy policy.
10. Calculate the SARSA target using the selected next action.
11. Update the Q-value of the current state-action pair.
12. Move to the next state and action.
13. Continue until the episode terminates or reaches the maximum number of steps.
14. Store the episode reward.
15. Decrease epsilon after each episode while maintaining the minimum epsilon value.
16. Repeat the process for all training episodes.
17. Derive the state-value function from the learned Q-table.
18. Derive the learned policy by selecting the action with the highest Q-value for each state.
19. Calculate the average reward over the last 1000 episodes.
20. Plot the moving average of episode rewards to visualize the learning progress.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    desc=[
        "FFSF",
        "FFFF",
        "FFFF",
        "FFGF"
    ],
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

# Variable epsilon
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((num_states, num_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        # Exploration
        return env.action_space.sample()

    else:
        # Exploitation
        return np.argmax(Q[state])


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Environment decides the initial state
    state, info = env.reset()

    # Select first action using current epsilon
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # If episode has not ended,
        # select next action using current policy
        if not terminated and not truncated:

            next_action = epsilon_greedy_action(
                next_state,
                epsilon
            )

            # SARSA target
            td_target = (
                reward
                + gamma * Q[next_state, next_action]
            )

        else:

            # No next action for terminal state
            next_action = None

            td_target = reward

        # SARSA update
        td_error = td_target - Q[state, action]

        Q[state, action] = (
            Q[state, action]
            + alpha * td_error
        )

        # Stop if episode is finished
        if terminated or truncated:
            break

        # Move to next state and action
        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # -------------------------------------------------
    # Variable Epsilon Decay
    # -------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# State-Value Function and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(Q, 3)
)

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)

print(
    "\nFinal epsilon:",
    epsilon
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "SARSA Learning Curve - FrozenLake"
)

plt.grid(True)

plt.show()

env.close()
```

---

## Output

```text
Final Q-table:

[[0.796 0.874 0.827 0.792]
 [0.876 0.870 0.877 0.862]
 [0.880 0.888 0.880 0.878]
 [0.882 0.880 0.880 0.878]
 [0.875 0.888 0.872 0.871]
 [0.888 0.900 0.889 0.884]
 [0.898 0.914 0.897 0.892]
 [0.896 0.911 0.899 0.898]
 [0.895 0.904 0.895 0.900]
 [0.905 0.931 0.912 0.903]
 [0.926 0.952 0.932 0.934]
 [0.940 0.950 0.939 0.936]
 [0.903 0.902 0.903 0.913]
 [0.920 0.929 0.958 0.926]
 [0.000 0.000 0.000 0.000]
 [0.965 0.980 0.961 0.965]]

Estimated State-Value Function:

[[0.874 0.877 0.888 0.882]
 [0.888 0.900 0.914 0.911]
 [0.904 0.931 0.952 0.950]
 [0.913 0.958 0.000 0.980]]

Learned Policy:

[['D' 'R' 'D' 'L']
 ['D' 'D' 'D' 'D']
 ['D' 'D' 'D' 'D']
 ['U' 'R' 'L' 'D']]

Average reward over last 1000 episodes: 1.0

Final epsilon: 0.05
```

---

## Result

```text
The SARSA control algorithm was successfully implemented using the
Gymnasium FrozenLake-v1 environment.

The agent learned an action-value function through repeated
interaction with the environment using an epsilon-greedy policy.

The epsilon value was gradually reduced from 1.0 to a minimum of
0.05, allowing the agent to shift from exploration to exploitation.

The learned Q-table was used to derive the state-value function and
the final policy.

The average reward over the last 1000 episodes was 1.0, indicating
that the agent successfully learned a policy capable of reaching
the goal consistently during the final stages of training.
```

---

## Inference

```text
The experiment demonstrates that SARSA can learn an effective policy
through on-policy temporal-difference learning.

Initially, the agent explores different actions using a high epsilon
value. As training progresses, epsilon decreases and the agent relies
more on the learned Q-values.

The final Q-table represents the estimated quality of different
actions in each state. The learned policy selects the action with the
highest Q-value for each state.

The learning curve and average reward show that the agent improves
its performance through repeated interaction with the environment.
```

---

## Conclusion

The SARSA control algorithm was successfully implemented using
Gymnasium's FrozenLake environment. The agent learned an effective
action-value function and improved its policy through repeated
interaction with the environment. The use of variable epsilon
provided a balance between exploration and exploitation, allowing the
agent to discover useful actions initially and exploit the learned
knowledge during later episodes.
