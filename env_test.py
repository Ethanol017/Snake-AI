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
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    if reward != old_reward:
        print("reward", reward)
        old_reward = reward
    if terminated or truncated:
        break
    state = next_state
    env.render()
env.close()


