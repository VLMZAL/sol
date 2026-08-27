import numpy as np
from qnn import qnn_policy
from env import Env

# Carica i pesi esistenti se ci sono
try:
    weights = np.load("weights.npy")
    print("Pesi caricati:", weights)
except:
    weights = np.random.uniform(-1, 1, size=3)
    print("Nessun file trovato, creati nuovi pesi:", weights)

# Learning rate molto basso per 10k episodi
lr = 0.0001

EPISODES = 10000
MAX_STEPS = 50

env = Env()

last_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):
        p = qnn_policy(state, weights)
        action = int((p + 1) * 2) % 4

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Gradiente REINFORCE semplice
        grad = reward * (state - 0.5)
        weights += lr * grad

        # Clipping per evitare esplosione dei pesi
        weights = np.clip(weights, -5, 5)

        state = next_state
        if done:
            break

    # Tracking reward
    last_rewards.append(total_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)

    # Stampa ogni 100 episodi
    if episode % 100 == 0:
        print(f"Episodio {episode} | Reward medio ultimi 100: {np.mean(last_rewards):.3f}")

# Salva i pesi aggiornati
np.save("weights.npy", weights)
print("Pesi aggiornati e salvati:", weights)
