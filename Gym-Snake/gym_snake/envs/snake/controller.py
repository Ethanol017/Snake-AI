from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np


class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30, 30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True, step_limit=150):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)

        self.snakes = []
        self.dead_snakes = []
        self.snake_sizes = []
        for i in range(1, n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            self.snakes.append(Snake(start_coord, snake_size))
            # self.snake_sizes[i] = self
            color = [self.grid.HEAD_COLOR[0], i*10, 0]
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2, n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.food_coord = self.grid.new_food()

        self.step_limit = step_limit
        self.steps_without_food = 0
        
        self.prev_distance = abs(start_coord[0]-self.food_coord[0]) + abs(start_coord[1]-self.food_coord[1])
        self.current_distance = self.prev_distance

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            # Avoid miscount of grid.open_space
            self.grid.cover(snake.head, snake.head_color)
            self.grid.connect(snake.body.popleft(),
                              snake.body[0], self.grid.SPACE_COLOR)
            self.kill_snake(snake_idx)
            reward = -10
            return reward
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.steps_without_food = 0
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR)  # Redraw tail
            self.grid.connect(
                snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            # Avoid miscount of grid.open_space
            self.grid.cover(snake.head, snake.head_color)
            reward = 10
            self.food_coord = self.grid.new_food()
            self.step_limit *= 1.1
        else:
            reward = 0
            empty_coord = snake.body.popleft()
            self.grid.connect(
                empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """

        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """
        # print(directions)
        self.steps_without_food += 1
        truncated = self.steps_without_food >= self.step_limit

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if isinstance(directions, (int, np.integer)) or len(directions) == 1:
                return self.grid.grid.copy(), 0, True, truncated, {"snakes_remaining": self.snakes_remaining}
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, truncated, {"snakes_remaining": self.snakes_remaining}

        rewards = []

        if isinstance(directions, (int, np.integer)):
            directions = [directions]
        for i, direction in enumerate(directions):
            if not truncated and self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction, i)
            if type(self.snakes[i]) != type(None):
                self.current_distance = abs(self.snakes[i].head[0]-self.food_coord[0]) + abs(self.snakes[i].head[1]-self.food_coord[1])
            
            # reward
            if truncated:
                reward = -10
            else:
                reward = self.move_result(direction, i)
                
            if reward > 0:
                self.current_distance = abs(self.snakes[i].head[0]-self.food_coord[0]) + abs(self.snakes[i].head[1]-self.food_coord[1])
                reward += len(self.snakes[i].body) * 0.5
            elif reward != -5:
                dis = (self.prev_distance - self.current_distance)
                reward += 0.2 if dis > 0 else -0.1
                reward -= 0.01 * (1.01**self.steps_without_food)
            rewards.append(reward)

            self.prev_distance = self.current_distance    

        terminated = self.snakes_remaining < 1 or self.grid.open_space < 1

        # done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) == 1:
            return self.grid.grid.copy(), rewards[0], terminated, truncated, {"snakes_remaining": self.snakes_remaining}
        else:
            return self.grid.grid.copy(), rewards, terminated, truncated, {"snakes_remaining": self.snakes_remaining}
