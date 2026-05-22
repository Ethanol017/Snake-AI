import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from model import SnakeDQN
import gym_snake  # type: ignore

MODEL_PATH = "models/snake_dqn_step60000000.pth"
GAMES = 100
MODELS_DIR = "models"
MAX_SIZE_STATE_PATH = "./max_size_state.npy"


def process_obs(observation, device, one_hot_lut):
    obs = np.asarray(observation, dtype=np.int64)
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape (H, W), got {obs.shape}")

    obs_indices = torch.as_tensor(obs, dtype=torch.long, device=device)
    one_hot = one_hot_lut[obs_indices]  # (H, W, C)
    return one_hot.permute(2, 0, 1).unsqueeze(0).contiguous()


def resolve_dqn_checkpoint(model_path=None, models_dir="models"):
    if model_path:
        resolved_path = Path(model_path).expanduser()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
        return resolved_path

    candidates = []
    for path in Path(models_dir).glob("snake_dqn_step*.pth"):
        match = re.search(r"snake_dqn_step(\d+)\.pth$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No DQN checkpoints found under {Path(models_dir).resolve()}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("snake-v0")
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)

    checkpoint_path = resolve_dqn_checkpoint(model_path=MODEL_PATH, models_dir=MODELS_DIR)
    model = SnakeDQN(channel=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    sizes = []
    best_size = -1
    best_state = None
    try:
        for episode in range(GAMES):
            obs, _ = env.reset()
            done = False

            while not done:
                obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
                with torch.no_grad():
                    q_values = model(obs_tensor)
                action = int(torch.argmax(q_values, dim=-1).item())

                obs, reward, terminated, truncated, info = env.step(action)
                next_obs_tensor = process_obs(obs, device=device, one_hot_lut=one_hot_lut)
                done = bool(np.logical_or(terminated, truncated))
                episode_size = int(np.asarray(info.get("snake_size", 0)).reshape(-1)[0])
                if episode_size > best_size:
                    best_size = episode_size
                    best_state = next_obs_tensor.detach().cpu().numpy()

            sizes.append(episode_size)
            print(f"episode: {episode + 1} size: {episode_size} best_size: {max(sizes)}")
    finally:
        env.close()

    if best_state is not None:
        np.save(MAX_SIZE_STATE_PATH, best_state)
        print(f"Saved max-size state to {MAX_SIZE_STATE_PATH}")

    print(f"Model: {checkpoint_path}")
    print(f"Mean size: {np.mean(sizes)}")
    print(f"Std size: {np.std(sizes)}")


if __name__ == "__main__":
    main()
