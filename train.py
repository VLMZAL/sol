import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from qnn import qnn_policy
from env import Env

# Carica i pesi esistenti se ci sono
try:
    weights = np.array(onp.load("weights.npy"), requires_grad=True)
    print("Pesi caricati:", weights)
except FileNotFoundError:
    weights = np.array(onp.random.uniform(-1, 1, size=3), requires_grad=True)
    print("Nessun file trovato, creati nuovi pesi:", weights)

# Learning rate molto basso per 10k episodi
lr = 0.0001
gamma = 0.99
epsilon = 1e-10

EPISODES = 10000
MAX_STEPS = 50

env = Env()

last_rewards = []

def trajectory_loss(current_weights, states, actions, returns):
    loss = 0.0
    for state, action, discounted_return in zip(states, actions, returns):
        probabilities = qnn_policy(state, current_weights)
        loss -= discounted_return * np.log(probabilities[action] + epsilon)
    return loss

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    states = []
    actions = []
    rewards = []

    for _ in range(MAX_STEPS):
        probabilities = onp.asarray(qnn_policy(state, weights), dtype=float)
        probabilities /= probabilities.sum()
        action = onp.random.choice(4, p=probabilities)

        next_state, reward, done = env.step(action)
        total_reward += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        if done:
            break

    returns = []
    discounted_return = 0.0
    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.insert(0, discounted_return)

    loss = lambda current_weights: trajectory_loss(
        current_weights, states, actions, returns
    )
    gradient = qml.grad(loss)(weights)
    weights = np.clip(weights - lr * gradient, -5, 5)

    # Tracking reward
    last_rewards.append(total_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)

    # Stampa ogni 100 episodi
    if episode % 100 == 0:
        print(f"Episodio {episode} | Reward medio ultimi 100: {np.mean(last_rewards):.3f}")

# Salva i pesi aggiornati
onp.save("weights.npy", onp.asarray(weights))
print("Pesi aggiornati e salvati:", weights)
