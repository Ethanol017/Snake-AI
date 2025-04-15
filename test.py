from main import test
import gymnasium as gym
import torch
from model import SnakePPO

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('snake-v0')
    model = SnakePPO().to(device)
    test(model=model,model_name='./models/snake_ppo_7000.pth',env=env,test_times=10,render=True)