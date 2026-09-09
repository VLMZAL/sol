import numpy as np

WIDTH = 600
HEIGHT = 400

class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_x = np.random.randint(0, WIDTH)
        self.player_y = np.random.randint(0, HEIGHT)
        self.target_x = np.random.randint(0, WIDTH)
        self.target_y = np.random.randint(0, HEIGHT)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        dx = (self.target_x - self.player_x) / WIDTH
        dy = (self.target_y - self.player_y) / HEIGHT
        return np.array([
            self.player_x / WIDTH,
            self.player_y / HEIGHT,
            dx,
            dy
        ])

    def step(self, action):
        # 0=dx+, 1=dx-, 2=dy+, 3=dy-
        if action == 0:
            self.player_x += 5
        elif action == 1:
            self.player_x -= 5
        elif action == 2:
            self.player_y += 5
        elif action == 3:
            self.player_y -= 5

        # Limiti schermo
        self.player_x = np.clip(self.player_x, 0, WIDTH)
        self.player_y = np.clip(self.player_y, 0, HEIGHT)

        dist = np.sqrt((self.player_x - self.target_x)**2 + (self.player_y - self.target_y)**2)
        reached = dist < 20

        # Reward forte per far imparare la QNN
        if reached:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        self.steps += 1
        if self.steps >= 50:
            done = True

        return self.get_state(), reward, done
