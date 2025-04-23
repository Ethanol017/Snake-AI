import os
import time
import gymnasium as gym
import numpy as np
import torch
from model import SnakePPO
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import gym_snake # type: ignore



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fast_downsample(observation, cell_size=10):
    result = observation[cell_size//2::cell_size, cell_size//2::cell_size, :]
    return result

def train(model,env,episodes,epochs,buffer_size,batch_size,lr,gamma,lambda_,clip_ppo,lr_gamma,c1=0.5,c2=0.01,save_interval=100,save_dir='./models/',log_dir='./logs/'):
    from torch.utils.tensorboard import SummaryWriter
    current_time = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, f"snake_ppo_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=lr_gamma)
    
    buffer_actions = []
    buffer_rewards = []
    buffer_dones = []
    buffer_log_probs = []
    buffer_values = []
    buffer_states = []
    play_count = 0
    episode_count = 0
    while episode_count < episodes:
        # play game
        state,_ = env.reset()
        state = fast_downsample(state)
        state_tensor = torch.from_numpy(state).permute(2, 0, 1).float().to(device)
        done = False
        reward_sum = 0
        snake_size = 0
        
        for step in range(buffer_size):
            with torch.no_grad():
                action_prob, state_value = model(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(action_prob)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                snake_size = info['snake_size']
                done = terminated or truncated
                reward_sum += reward
                next_state_tensor = torch.from_numpy(fast_downsample(next_state)).permute(2, 0, 1).float().to(device)
                
                buffer_states.append(state_tensor.detach().cpu())
                buffer_actions.append(action.detach().cpu())
                buffer_rewards.append(reward)
                buffer_dones.append(done)
                buffer_log_probs.append(log_prob.detach().cpu())
                buffer_values.append(state_value.squeeze().detach().cpu())
                
                state_tensor = next_state_tensor
                
                if done:            
                    state, _ = env.reset()
                    state = fast_downsample(state)
                    state_tensor = torch.from_numpy(state).permute(2, 0, 1).float().to(device)
                    writer.add_scalar("gameplay/reward", reward_sum, play_count)
                    writer.add_scalar("gameplay/snake_size", snake_size, play_count)
                    reward_sum = 0
                    snake_size = 0
                    play_count += 1
                    if episode_count >= episodes: 
                        break 
  
        # --train--
        if episode_count < episodes:
            episode_count += 1
            # compute_advantages
            b_advantages = torch.zeros(buffer_size).to(device)
            with torch.no_grad():
                _ , last_value = model(state_tensor.unsqueeze(0))
                last_value = last_value.squeeze(-1)
            
            last_adv = 0
            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_value = last_value
                else:
                    next_value = buffer_values[t + 1]
                
                delta = buffer_rewards[t] + gamma * next_value * (1 - int(buffer_dones[t])) - buffer_values[t]
                last_adv = delta + gamma * lambda_ * (1 - int(buffer_dones[t])) * last_adv
                b_advantages[t] = last_adv
            b_returns = b_advantages + torch.stack(buffer_values).to(device)
            
            
            b_states = torch.stack(buffer_states).to(device)
            b_actions = torch.stack(buffer_actions).squeeze().to(device)
            b_log_probs = torch.stack(buffer_log_probs).squeeze().to(device)
            
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # print(f"b_advantages: {b_advantages.shape}")
            
            indices = np.arange(buffer_size)
            for epoch in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, buffer_size, batch_size):
                    mb_indices = indices[start : start + batch_size]

                    mb_states = b_states[mb_indices]
                    mb_actions = b_actions[mb_indices]
                    mb_old_log_probs = b_log_probs[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_returns = b_returns[mb_indices]

                    new_action_probs, current_values = model(mb_states)
                    dist = torch.distributions.Categorical(new_action_probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = (new_log_probs - mb_old_log_probs).exp()

                    # Policy Loss
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_ppo, 1 + clip_ppo) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    current_values = current_values.squeeze()
                    value_loss = F.mse_loss(current_values, mb_returns)
                    # Total Loss
                    loss = policy_loss + c1 * value_loss - c2 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
            scheduler.step()
            
            writer.add_scalar("loss/policy_loss", policy_loss.item(), episode_count)
            writer.add_scalar("loss/value_loss", value_loss.item(), episode_count)
            writer.add_scalar("loss/entropy", entropy.item(), episode_count)
            writer.add_scalar("loss/loss", loss.item(), episode_count)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], episode_count)
            print(f"Episode {episode_count}/{episodes} , PlayCount {play_count}")
        
        buffer_states.clear()
        buffer_actions.clear()
        buffer_rewards.clear()
        buffer_dones.clear()
        buffer_log_probs.clear()
        buffer_values.clear()

        if episode_count > 0 and episode_count % save_interval == 0:
            save_path = os.path.join(save_dir, f"snake_ppo_ep{episode_count}.pth")
            torch.save(model.state_dict(), save_path)
            
def test(model,model_name,env,test_times=10,render=False):
    model.load_state_dict(torch.load(model_name,weights_only=True,map_location=device))
    model.eval()
    for i in range(test_times):
        state, _ = env.reset()
        state_tensor = torch.from_numpy(fast_downsample(state)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        snake_length = 0
        reward_sum = 0
        while True:
            action_prob, _ = model(state_tensor)
            action = action_prob.argmax().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(reward)
            # print(info["snake_size"])
            reward_sum += reward
            state_tensor = torch.from_numpy(fast_downsample(next_state)).permute(2, 0, 1).float().unsqueeze(0).to(device)
            if render:
                env.render()
            
            if terminated or truncated:
                print(f"Test {i+1}/{test_times}, Snake Reward: {reward_sum}")
                break
            
            
if __name__ == '__main__':
    env = gym.make('snake-v0')
    model = SnakePPO().to(device)
    
    train(
        model=model,
        env=env,
        episodes = 20000,
        epochs=10,
        buffer_size=1024,
        batch_size=128,
        gamma=0.99,
        lambda_=0.95,
        lr=5e-4,
        clip_ppo=0.2,
        c1=0.5,
        c2=0.02,
        save_interval=1000,
        lr_gamma=0.99,
        save_dir = './models/',
        log_dir = './logs/'
    )
    test(model=model,model_name='./models/snake_ppo_300.pth',env=env,test_times=10,render=True)
    env.close()
    
    
