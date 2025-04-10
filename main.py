from collections import deque
import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from model import SnakePPO
import gym_snake # type: ignore

from torch.utils.tensorboard import SummaryWriter
current_time = time.strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join("logs", f"snake_ppo_{current_time}")
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,env,episodes,epochs,buffer_size,batch_size,lr,gamma,lambda_,epsilon_ppo,c1=0.5,c2=0.01,save_interval=100):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    recent_rewards = deque(maxlen=100)
    replay_buffer = []
    for episode in range(1,episodes+1):
        # play game
        state,_ = env.reset()
        state_tensor = torch.tensor(np.transpose(state, (2, 0, 1)),dtype=torch.float32).to(device)
        done = False
        reward_sum = 0
        while not done:
            with torch.no_grad():
                action_prob, state_value = model(state_tensor.unsqueeze(0))
                action = torch.multinomial(action_prob,1)
                log_prob = torch.distributions.Categorical(action_prob).log_prob(action)
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            next_state_tensor = torch.tensor(np.transpose(next_state, (2, 0, 1)),dtype=torch.float32).to(device)

            replay_buffer.append((
                state_tensor.cpu().numpy(),
                next_state_tensor.cpu().numpy(),
                action.item(),
                log_prob.item(),
                reward,
                state_value.item(),
                done
            ))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)
                
            state_tensor = next_state_tensor
        
        recent_rewards.append(reward_sum)
        if len(replay_buffer) >= batch_size:
            # --train--
            
            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in indices]
            states = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float).to(device)
            next_states = torch.tensor(np.stack([t[1] for t in batch]), dtype=torch.float).to(device)
            actions = torch.tensor([t[2] for t in batch], dtype=torch.long).to(device)
            old_log_probs = torch.tensor([t[3] for t in batch], dtype=torch.float).to(device)
            rewards = torch.tensor([t[4] for t in batch], dtype=torch.float).to(device)
            old_state_values = torch.tensor([t[5] for t in batch], dtype=torch.float).to(device)
            dones = torch.tensor([t[6] for t in batch], dtype=torch.float).to(device)
            
            
            # compute_advantages
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            
            with torch.no_grad():
                # print(next_states.shape)
                _, next_values = model(next_states)
                next_values = next_values.squeeze()
                next_values = next_values * (1 - dones)
                
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - old_state_values[t]
                advantages[t] = delta + gamma * lambda_ * last_advantage * (1 - dones[t])
                
                last_advantage = advantages[t]
            
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + old_state_values
            
            for epoch in range(epochs):
                # compute ratio
                new_action_probs, state_values = model(states)
                dist = torch.distributions.Categorical(new_action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - old_log_probs).exp()
                
                # compute surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - epsilon_ppo, 1 + epsilon_ppo) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # compute value loss
                value_loss = ((returns - state_values.squeeze()) ** 2).mean()
                
                # update model
                optimizer.zero_grad()
                loss = policy_loss + c1 * value_loss - c2 * entropy
                loss.backward()
                optimizer.step()
                # --log--
                if epoch == epochs - 1:
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), episode)
                    writer.add_scalar("loss/value_loss", value_loss.item(), episode)
                    writer.add_scalar("loss/loss", loss.item(), episode)
                    writer.add_scalar("reward", rewards.sum(), episode)
                    writer.add_scalar("advantage", advantages.mean(), episode)
                
                if episode % save_interval == 0:
                    print(f"Episode {episode}/{episodes}, Loss: {loss.item()}, Reward: {rewards.sum().item()}")
                    torch.save(model.state_dict(), f"./models/snake_ppo_{episode}.pth")
        
        scheduler.step(np.mean(recent_rewards))
            
def test(model,model_name,env,test_times=10,render=False):
    model.load_state_dict(torch.load(model_name,weights_only=True))
    for i in range(test_times):
        state, _ = env.reset()
        state = np.transpose(state, (2, 0, 1))
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        snake_length = 0
        while True:
            action_prob, _ = model(state_tensor)
            action = action_prob.argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            if reward == 1:
                snake_length += 1
            state_tensor = torch.tensor(np.transpose(next_state, (2, 0, 1)), dtype=torch.float).unsqueeze(0).to(device)
            
            if render:
                env.render()
            
            if terminated or truncated:
                print(f"Test {i+1}/{test_times}, Snake Length: {snake_length}")
                break
            
            
if __name__ == '__main__':
    env = gym.make('snake-v0')
    model = SnakePPO().to(device)
    
    train(
        model=model,
        env=env,
        episodes=3000,          
        epochs=4,
        buffer_size=2048,               
        batch_size=256,         
        gamma=0.99,             
        lambda_=0.95,           
        lr=1e-4,                
        epsilon_ppo=0.2,        
        c1=0.5,                 
        c2=0.02,                
        save_interval=100
    )    
    test(model=model,model_name='./models/snake_ppo_1300.pth',env=env,test_times=1,render=True)
    env.close()
    
    
