import pygame
import numpy as np
from qnn import qnn_policy
from env import Env

weights = np.load("weights.npy")

pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

env = Env()
state = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # QNN action probabilities
    probabilities = qnn_policy(state, weights)

    # Azione più lenta e più chiara
    action = int(np.argmax(probabilities))

    # Step dell'ambiente
    state, reward, done = env.step(action)

    if done:
        state = env.reset()

    # --- GRAFICA ---
    screen.fill((20, 20, 20))

    # Player
    pygame.draw.circle(screen, (0, 200, 255), (env.player_x, env.player_y), 12)

    # Target (si illumina se vicino)
    dist = np.sqrt((env.player_x - env.target_x)**2 + (env.player_y - env.target_y)**2)
    color = (255, 50, 50) if dist > 20 else (255, 200, 0)
    pygame.draw.circle(screen, color, (env.target_x, env.target_y), 12)

    # HUD informativo
    hud_lines = [
        f"QNN probabilities: {np.asarray(probabilities).round(3)}",
        f"Azione: {action}",
        f"Distanza: {dist:.1f}",
        f"Reward: {reward:.2f}"
    ]

    y_offset = 10
    for line in hud_lines:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y_offset))
        y_offset += 22

    pygame.display.flip()

    # FPS più lento per capire cosa succede
    clock.tick(15)

pygame.quit()
