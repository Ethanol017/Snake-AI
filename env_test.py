import time
import gymnasium as gym
import numpy as np
import gym_snake # type: ignore

# start = time.time()

# num_envs = 1
# envs = gym.make_vec('snake-v0',num_envs,vectorization_mode="async")

# observation, _ = envs.reset()
# print(envs.observation_space.shape)
# for i in range(100):
#     # envs.render()
#     actions = envs.action_space.sample()
#     obs, rewards, terminated, truncated, infos = envs.step(actions)

# envs.close()

# end = time.time()
# print("time", end - start)

env = gym.make("snake-v0")


state, _ = env.reset()
old_reward = -100
while True:
    # action = env.action_space.sample()
    # 0: up
    # 1: right
    # 2: down
    # 3: left
    action = int(input())
    next_state, reward, terminated, truncated, _ = env.step(action)
    cell_size = 10
    print(next_state[cell_size//2::cell_size, cell_size//2::cell_size, :])
    # if reward != old_reward:
    #     print("reward", reward)
    #     old_reward = reward
    print(terminated,truncated,reward)
    if terminated or truncated:
        break
    state = next_state
    env.render()
env.close()


