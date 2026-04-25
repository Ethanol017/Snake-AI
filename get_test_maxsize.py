import gymnasium as gym
import numpy as np
import torch
from model import SnakePPO
import gym_snake  # type: ignore

def process_obs(observation, device, one_hot_lut):
    obs = np.asarray(observation, dtype=np.int64)
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape (H, W), got {obs.shape}")

    obs_indices = torch.as_tensor(obs, dtype=torch.long, device=device)
    one_hot = one_hot_lut[obs_indices]  # (H, W, C)
    return one_hot.permute(2, 0, 1).unsqueeze(0).contiguous()

if __name__ == "__main__":
    
    model_path = "models/snake_ppo_ep2000.pth"
    game_times = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("snake-v0")
    model = SnakePPO(channel=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)

    obs, _ = env.reset()
    obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
    
    best_size = 0
    for episode in range(game_times):
        done = False
        episode_reward = 0.0

        while not done:
            obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
            with torch.no_grad():
                action_logits, _ = model(obs_tensor)
            action = int(torch.argmax(action_logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
        
        env.reset()
        
        if info.get('snake_size', 0) > best_size:
            best_size = info.get('snake_size', 0)
            np.save(f"max_size_state.npy", arr=obs_tensor.cpu().numpy())
        
        print(f"episode: {episode + 1} size: {info.get('snake_size', 0)} best_size: {best_size}")