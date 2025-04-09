import os, subprocess, time, signal
import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gym_snake.envs.snake import Controller, Discrete

try:
    import google.colab # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
try:
    import matplotlib.pyplot as plt
    import matplotlib
    if IN_COLAB:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    matplotlib.use('Agg')
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'],'render_fps':10}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.random_init = random_init

        self.action_space = spaces.Discrete(4)

        controller = Controller(
            self.grid_size, self.unit_size, self.unit_gap,
            self.snake_size, self.n_snakes, self.n_foods,
            random_init=self.random_init)
        grid = controller.grid
        self.observation_space = spaces.Box(
            low=np.min(grid.COLORS),
            high=np.max(grid.COLORS),
            dtype=np.uint8,
            shape=grid.grid.shape
        )

    def step(self, action):
        self.last_obs, rewards, terminated, truncated, info = self.controller.step(action)
        return self.last_obs, rewards, terminated, truncated , info

    def reset(self,seed=None,options=None):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        
        return self.last_obs,{}

    def render(self, mode='human', close=False, frame_speed=.1):
        if IN_COLAB:
            from IPython.display import display, clear_output # type: ignore
            if self.viewer is None:
                self.fig = plt.figure(figsize=(8, 8))
                self.viewer = self.fig.add_subplot(111)
            
                self.viewer.clear()
                self.viewer.imshow(self.last_obs)
                self.viewer.axis('off')
                
                clear_output(wait=True)
                display(plt.gcf())
        else:
            if self.viewer is None:
                self.fig = plt.figure()
                self.viewer = self.fig.add_subplot(111)
                plt.ion()
                self.fig.show()
            
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
            self.fig.canvas.draw()

    def seed(self, x):
        pass
