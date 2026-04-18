import gymnasium as gym
import gym_snake
import time

if __name__ == "__main__":
    print("PRESS q to quit")
    print("wasd to move, f to press")
    env = gym.make("snake-v0", render_mode="human")

    rew = 0
    obs, _ = env.reset()
    key = ""
    action = "w"
    while key != "q":
        env.render()
        key = input("action: ")
        if key == "w":
            action = 0
        elif key == "d":
            action = 1
        elif key == "s":
            action = 2
        elif key == "a":
            action = 3
        else:
            pass
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("rew:", rew)
        print("done:", done)
        print("info")
        for k in info.keys():
            print("    ", k, ":", info[k])
        if done:
            obs, _ = env.reset()
