from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class SnakeCore:
    """High-throughput single-snake game core with O(1) food sampling."""

    EMPTY = 0
    BODY = 1
    HEAD = 2
    FOOD = 3

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    _DELTAS = np.asarray(
        [
            [0, -1],
            [1, 0],
            [0, 1],
            [-1, 0],
        ],
        dtype=np.int16,
    )
    
    REWARD_FOOD = 15.0
    REWARD_STEP = -0.1
    REWARD_DEATH = -10.0
    REWARD_FILLED = 100.0
    REWARD_PBRS_GAMMA = 0.96
    REWARD_PBRS_COEFF = 1.0

    def __init__(
        self,
        width: int,
        height: int,
        init_length: int,
        step_limit: int,
        random_init: bool,
        rng: np.random.Generator,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("grid_size must be at least 3x3")
        if init_length < 2:
            raise ValueError("snake_size must be at least 2")
        if init_length > width:
            raise ValueError("snake_size must be <= grid width")
        if step_limit < 1:
            raise ValueError("step_limit must be >= 1")

        self.width = int(width)
        self.height = int(height)
        self.max_cells = self.width * self.height
        self.init_length = int(init_length)
        self.step_limit = int(step_limit)
        self.random_init = bool(random_init)
        self._rng = rng

        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

        self._x_buffer = np.empty(self.max_cells, dtype=np.int16)
        self._y_buffer = np.empty(self.max_cells, dtype=np.int16)

        self._all_cells = np.arange(self.max_cells, dtype=np.int32)
        self._free_cells = np.empty(self.max_cells, dtype=np.int32)
        self._free_pos = np.empty(self.max_cells, dtype=np.int32)
        self._free_count = 0

        self.head_idx = 0
        self.tail_idx = 0
        self.length = self.init_length
        self.direction = self.RIGHT

        self.food_x = -1
        self.food_y = -1
        self.max_distance = float((self.width - 1) + (self.height - 1))
        self.prev_potential = 0.0
        self.steps_without_food = 0
        self.terminated = False
        self.truncated = False

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value

    def reset(self) -> Tuple[np.ndarray, Dict[str, int]]:
        self.board.fill(self.EMPTY)

        self._free_cells[:] = self._all_cells
        self._free_pos[:] = self._all_cells
        self._free_count = self.max_cells

        self.length = self.init_length
        self.direction = self.RIGHT
        self.steps_without_food = 0
        self.terminated = False
        self.truncated = False

        if self.random_init:
            head_x = int(self._rng.integers(self.init_length - 1, self.width))
            head_y = int(self._rng.integers(0, self.height))
        else:
            head_x = max(self.init_length - 1, self.width // 2)
            head_y = self.height // 2

        tail_x = head_x - (self.length - 1)
        tail_y = head_y

        self.tail_idx = 0
        for offset in range(self.length):
            x = tail_x + offset
            y = tail_y
            self._x_buffer[offset] = x
            self._y_buffer[offset] = y
            cell = self._index(x, y)
            self._remove_from_free(cell)
            self.board[y, x] = self.BODY if offset < self.length - 1 else self.HEAD
        self.head_idx = self.length - 1

        self._spawn_food()
        self.prev_potential = self._potential(self.head_x, self.head_y, self.food_x, self.food_y)

        return self.board.copy(), {"snake_size": int(self.length)}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, int]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on terminated or truncated game. Call reset() to start a new game.")

        action_int = int(action) % 4
        # Avoid 180 degree turns
        if abs(action_int - self.direction) != 2:
            self.direction = action_int

        self.steps_without_food += 1
        # Time out (truncated)
        if self.steps_without_food >= self.step_limit:
            return self._die(truncated=True)

        dx, dy = self._DELTAS[self.direction]
        next_x = self.head_x + int(dx)
        next_y = self.head_y + int(dy)

        if not (0 <= next_x < self.width and 0 <= next_y < self.height):
            return self._die(truncated=False)

        next_code = int(self.board[next_y, next_x])
        will_eat = next_code == self.FOOD

        tail_x = self.tail_x
        tail_y = self.tail_y
        next_cell = self._index(next_x, next_y)
        tail_cell = self._index(tail_x, tail_y)

        if next_code in (self.BODY, self.HEAD):
            if not (next_cell == tail_cell and not will_eat):
                return self._die(truncated=False)

        self.board[self.head_y, self.head_x] = self.BODY

        if not will_eat:
            self.board[tail_y, tail_x] = self.EMPTY
            self._add_to_free(tail_cell)
            self.tail_idx = (self.tail_idx + 1) % self.max_cells

        self.head_idx = (self.head_idx + 1) % self.max_cells
        self._x_buffer[self.head_idx] = next_x
        self._y_buffer[self.head_idx] = next_y

        # If eating, it has been removed from free, so don't call function again
        if not will_eat:
            self._remove_from_free(next_cell)

        self.board[next_y, next_x] = self.HEAD

        reward = self.REWARD_STEP

        if will_eat:
            self.length += 1
            self.steps_without_food = 0
            reward += self.REWARD_FOOD

            spawned = self._spawn_food()
            if not spawned:
                self.terminated = True
                reward = self.REWARD_FILLED
            self.prev_potential = self._potential(self.head_x, self.head_y, self.food_x, self.food_y)
        else:
            current_potential = self._potential(self.head_x, self.head_y, self.food_x, self.food_y)
            reward += self.REWARD_PBRS_COEFF * (self.REWARD_PBRS_GAMMA * current_potential - self.prev_potential)
            self.prev_potential = current_potential

        return (
            self.board.copy(),
            float(reward),
            bool(self.terminated),
            False,
            {"snake_size": int(self.length)},
        )

    @property
    def head_x(self) -> int:
        return int(self._x_buffer[self.head_idx])

    @property
    def head_y(self) -> int:
        return int(self._y_buffer[self.head_idx])

    @property
    def tail_x(self) -> int:
        return int(self._x_buffer[self.tail_idx])

    @property
    def tail_y(self) -> int:
        return int(self._y_buffer[self.tail_idx])

    def _die(self, truncated: bool) -> Tuple[np.ndarray, float, bool, bool, Dict[str, int]]:
        self.terminated = True
        self.truncated = bool(truncated)
        reward = self.REWARD_DEATH
        # Apply terminal-state PBRS transition with Phi(terminal)=0.
        reward += self.REWARD_PBRS_COEFF * (0.0 - self.prev_potential)
        self.prev_potential = 0.0
        return (
            self.board.copy(),
            float(reward),
            True,
            bool(self.truncated),
            {"snake_size": int(self.length)},
        )

    def _spawn_food(self) -> bool:
        if self._free_count <= 0:
            self.food_x = -1
            self.food_y = -1
            return False

        choice_pos = int(self._rng.integers(0, self._free_count))
        cell = int(self._free_cells[choice_pos])
        self._remove_from_free(cell)

        self.food_x = int(cell % self.width)
        self.food_y = int(cell // self.width)
        self.board[self.food_y, self.food_x] = self.FOOD
        return True

    def _remove_from_free(self, cell: int) -> None:
        pos = int(self._free_pos[cell])
        if pos < 0 or pos >= self._free_count:
            return

        last_pos = self._free_count - 1
        last_cell = int(self._free_cells[last_pos])
        self._free_cells[pos] = last_cell
        self._free_pos[last_cell] = pos
        self._free_count = last_pos
        self._free_pos[cell] = -1

    def _add_to_free(self, cell: int) -> None:
        if int(self._free_pos[cell]) != -1:
            return

        self._free_cells[self._free_count] = cell
        self._free_pos[cell] = self._free_count
        self._free_count += 1

    def _index(self, x: int, y: int) -> int:
        return y * self.width + x

    @staticmethod
    def _distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _potential(self, x1: int, y1: int, x2: int, y2: int) -> float:
        distance = float(self._distance(x1, y1, x2, y2))
        return 1.0 - (distance / self.max_distance)
