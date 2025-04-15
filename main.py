from collections import deque
import math
import os
import random
import time
import gymnasium as gym
import torch
from model import SnakePPO
from torch.optim.lr_scheduler import StepLR
import gym_snake # type: ignore



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fast_downsample(observation, cell_size=10):
    result = observation[cell_size//2::cell_size, cell_size//2::cell_size, :]
    return result

def train(model,env,episodes,epochs,buffer_size,batch_size,segment_length,lr,gamma,lambda_,clip_ppo,exploration_decay_rate,lr_gamma,c1=0.5,c2=0.01,save_interval=100,save_dir='./models/',log_dir='./logs/'):
    from torch.utils.tensorboard import SummaryWriter
    current_time = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, f"snake_ppo_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=lr_gamma)
    trajectory_buffer = deque(maxlen=buffer_size)
    for episode in range(1,episodes+1):
        # play game
        state,_ = env.reset()
        state = fast_downsample(state)
        state_tensor = torch.from_numpy(state).permute(2, 0, 1).float().to(device)
        done = False
        reward_sum = 0
        replay_buffer = []
        log_snake_reward = 0
        log_snake_size = 0
        while not done:
            with torch.no_grad():
                action_prob, state_value = model(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(action_prob)

                exploration_prob = max(0.05, 0.5 * math.exp(-exploration_decay_rate * episode / episodes))

                if random.random() < exploration_prob:
                    action = torch.randint(0, 4, (1,)).to(device)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                log_snake_size = info['snake_size']
                done = terminated or truncated
                reward_sum += reward
                next_state_tensor = torch.from_numpy(fast_downsample(next_state)).permute(2, 0, 1).float().to(device)
                log_snake_reward += reward
                replay_buffer.append([
                    state_tensor.detach().cpu(), # tensor
                    next_state_tensor.detach().cpu(), # tensor
                    action.detach().cpu(), # tensor
                    log_prob.detach().cpu(), # tensor
                    reward, # float
                    state_value.squeeze(0).detach().cpu(), # tensor
                    done # bool
                ])
                state_tensor = next_state_tensor
        
        trajectory_buffer.append(replay_buffer)  #trajectory_buffer= [ [game_repaly] , [game_repaly] , [game_repaly], ... X N] ]

        if len(trajectory_buffer) >= batch_size:
            # --train--
            batch = []
            for i in range(batch_size//segment_length):
                ep = random.choice(trajectory_buffer)
                if len(ep) < segment_length:
                    pad_len = segment_length - len(ep)
                    last = ep[-1]
                    pad = [(last[0], last[1], torch.tensor([0]), torch.tensor([0]) ,0.0, torch.tensor([0]), torch.tensor(1.0))] * pad_len
                    batch.extend( ep+pad )
                else:
                    start_idx = random.randint(0, len(ep) - segment_length)
                    segment = ep[start_idx:start_idx + segment_length]
                    batch.extend(segment)

            #batch = [ [segment_length] , [segment_length] , ... X (batch_size//segment_length) ]
            states = torch.stack([t[0] for t in batch]).to(device)
            next_states = torch.stack([t[1] for t in batch]).to(device)
            actions = torch.stack([t[2] for t in batch]).to(device)
            old_log_probs = torch.stack([t[3] for t in batch]).to(device)
            rewards = torch.tensor([t[4] for t in batch],dtype=float).to(device)
            old_values = torch.tensor([t[5] for t in batch]).to(device)
            dones = torch.tensor([t[6] for t in batch],dtype=float).to(device)
            # compute_advantages
            advantages = torch.zeros_like(rewards)
            
            with torch.no_grad():
                # print(next_states.shape)
                _, next_values = model(next_states)
                next_values = next_values.squeeze(-1)
                next_values = next_values * (1 - dones)
                
                for seg_idx in range(batch_size // segment_length):
                    start = seg_idx * segment_length
                    end = (seg_idx + 1) * segment_length

                    last_gae  = 0
                    for t in reversed(range(start, end)):
                        if t == end - 1 or dones[t]:
                            next_value = 0
                        else:
                            next_value = next_values[t].item()
                            
                        delta = rewards[t] + gamma * next_value - old_values[t].item()
                        
                        last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
                        advantages[t] = last_gae
                
                mask = (dones != 1)
                valid_adv = advantages[mask]
                advantages[mask] = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
                returns = advantages + old_values.squeeze(-1)

            
            for epoch in range(epochs):
                # compute ratio
                new_action_probs, current_values = model(states)
                dist = torch.distributions.Categorical(new_action_probs)
                new_log_probs = dist.log_prob(actions.squeeze(-1))
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - old_log_probs.squeeze(-1)).exp()
                # compute surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_ppo, 1 + clip_ppo) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # compute value loss
                value_loss = ((current_values - returns) ** 2).mean()
                
                # update model
                optimizer.zero_grad()
                loss = policy_loss + c1 * value_loss - c2 * entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                # --log--
                if epoch == epochs - 1:
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), episode)
                    writer.add_scalar("loss/value_loss", value_loss.item(), episode)
                    writer.add_scalar("loss/loss", loss.item(), episode)
                    writer.add_scalar("reward", log_snake_reward , episode)
                    writer.add_scalar("snake_size", log_snake_size , episode)

                    if episode % save_interval == 0:
                        print(f"Episode {episode}/{episodes}, Loss: {loss.item()}, Reward: {rewards.sum().item()}")
                        torch.save(model.state_dict(), f"{save_dir}/snake_ppo_{episode}.pth")
        
            scheduler.step()
            
def test(model,model_name,env,test_times=10,render=False):
    model.load_state_dict(torch.load(model_name,weights_only=True,map_location=device))
    for i in range(test_times):
        state, _ = env.reset()
        state_tensor = torch.from_numpy(fast_downsample(state)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        snake_length = 0
        reward_sum = 0
        while True:
            action_prob, _ = model(state_tensor)
            action = action_prob.argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            print(reward)
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
        buffer_size=2048,               
        batch_size=512,    
        segment_length = 128,     
        gamma=0.99,             
        lambda_=0.9,           
        lr=1e-4,                
        clip_ppo=0.1,     
        exploration_decay_rate = 2,
        c1=0.3,                 
        c2=0.01,                
        save_interval=1000,
        save_dir = './models/',
        log_dir = './logs/'
    )    
    test(model=model,model_name='./models/snake_ppo_300.pth',env=env,test_times=10,render=True)
    env.close()
    
    
