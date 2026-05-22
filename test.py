import gymnasium as gym
import numpy as np
import torch
from model import SnakeDQN, SnakePPO
import gym_snake  # type: ignore

TEST_EPISODES = 10


def process_obs(observation, device, one_hot_lut):
    obs = np.asarray(observation, dtype=np.int64)
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape (H, W), got {obs.shape}")

    obs_indices = torch.as_tensor(obs, dtype=torch.long, device=device)
    one_hot = one_hot_lut[obs_indices]  # (H, W, C)
    return one_hot.permute(2, 0, 1).unsqueeze(0).contiguous()


def test_PPO():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("snake-v0", render_mode="human")
    model_path = "models/snake_ppo_ep2000.pth"
    model = SnakePPO(channel=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)

    size_sum = 0.0
    size_max = float("-inf")
    for episode in range(TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            env.render()
            obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
            with torch.no_grad():
                action_logits, _ = model(obs_tensor)
            action = int(torch.argmax(action_logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)

        print(
            f"episode: {episode + 1} reward: {episode_reward:.2f} snake_size: {info['snake_size']}"
        )
        size_sum += info["snake_size"]
        size_max = max(size_max, info["snake_size"])

    print(
        f"episodes: {TEST_EPISODES} reward(mean/max): {size_sum/TEST_EPISODES:.2f}/{size_max:.2f}"
    )
    env.close()


def test_DQN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("snake-v0", render_mode="human")
    model_path = "models/snake_dqn_step60000000.pth"
    model = SnakeDQN(channel=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)
    size_sum = 0.0
    size_max = float("-inf")
    for episode in range(TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            env.render()
            obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
            with torch.no_grad():
                q_values = model(obs_tensor)
            action = int(torch.argmax(q_values, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
        print(
            f"episode: {episode + 1} reward: {episode_reward:.2f} snake_size: {info['snake_size']}"
        )
        size_sum += info["snake_size"]
        size_max = max(size_max, info["snake_size"])
    print(
        f"episodes: {TEST_EPISODES} reward(mean/max): {size_sum/TEST_EPISODES:.2f}/{size_max:.2f}"
    )
    env.close()


if __name__ == "__main__":
    test_PPO()
    # test_DQN()
