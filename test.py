import os
from main import test
import gymnasium as gym
import torch
from model import SnakePPO

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('snake-v0')
    model = SnakePPO().to(device)
    # max_length = 0
    # max_average_length = 0
    # max_name = ''
    # max_average_name = ''
    # for model_name in os.listdir('./models/'):
    #     if model_name.endswith('.pth') and 'ep' in model_name:
    #         print(f"Testing {model_name}...")
    #         m , t = test(model=model,model_name=os.path.join('./models/',model_name),env=env,test_times=100,render=False,output=False)
    #         if m > max_length:
    #             max_length = m
    #             max_name = model_name
    #         if t > max_average_length:  
    #             max_average_length = t
    #             max_average_name = model_name
    # print(f"Max Length: {max_length} , Model Name: {max_name} \nMax Average Length: {max_average_length} , Model Name: {max_average_name}")    
    test(model=model,model_name='./models/snake_ppo_ep1500.pth',env=env,test_times=100,render=False,output=False)
    env.close()