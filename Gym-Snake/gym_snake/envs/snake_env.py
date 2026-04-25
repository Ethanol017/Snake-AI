from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_snake.envs.fast_core import SnakeCore


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size=(12, 12),
        snake_size=3,
        step_limit=150,
        random_init=True,
        render_mode=None,
        pixel_size=24,
    ):
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.snake_size = int(snake_size)
        self.step_limit = int(step_limit)
        self.random_init = bool(random_init)
        self.render_mode = render_mode
        self.pixel_size = int(pixel_size)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(self.grid_size[1], self.grid_size[0]),
            dtype=np.uint8,
        )

        self._core: Optional[SnakeCore] = None
        self.last_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._figure = None
        self._axes = None
        self._palette = np.asarray(
            [
                [22, 22, 22],
                [45, 160, 65],
                [230, 70, 70],
                [30, 130, 210],
            ],
            dtype=np.uint8,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._core is None:
            self._core = SnakeCore(
                width=self.grid_size[0],
                height=self.grid_size[1],
                init_length=self.snake_size,
                step_limit=self.step_limit,
                random_init=self.random_init,
                rng=self.np_random,
            )
        else:
            self._core.rng = self.np_random

        self.last_obs, info = self._core.reset()
        return self.last_obs, info

    def step(self, action):
        if self._core is None:
            raise RuntimeError("Call reset() before step().")

        self.last_obs, reward, terminated, truncated, info = self._core.step(action)
        return self.last_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        rgb = self._to_rgb(self.last_obs)
        if self.render_mode == "rgb_array":
            return rgb

        if self.render_mode != "human":
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        import matplotlib.pyplot as plt

        if self._figure is None or self._axes is None:
            self._figure, self._axes = plt.subplots()
            plt.ion()

        self._axes.clear()
        self._axes.imshow(rgb)
        self._axes.axis("off")
        self._figure.canvas.draw_idle()
        plt.pause(1.0 / self.metadata["render_fps"])
        return None

    def close(self):
        if self._figure is None:
            return

        import matplotlib.pyplot as plt

        plt.close(self._figure)
        self._figure = None
        self._axes = None

    def _to_rgb(self, board: np.ndarray) -> np.ndarray:
        rgb = self._palette[board]
        if self.pixel_size <= 1:
            return rgb
        return np.repeat(
            np.repeat(rgb, self.pixel_size, axis=0), self.pixel_size, axis=1
        )
