import os
import time
import gymnasium as gym
import numpy as np
import torch
from model import SnakePPO
import torch.nn.functional as F
import gym_snake  # type: ignore



def _process_obs(observation, one_hot_lut, device):
    # Convert env obs to One-Hot encoding and add batch dim if needed
    obs = np.asarray(observation, dtype=np.int64)

    # Single env
    if obs.ndim == 2:
        obs = obs[None, ...]

    # Discrete map (B, H, W) -> one-hot (B, C, H, W)
    obs_indices = torch.as_tensor(obs, dtype=torch.long, device=device)
    one_hot = one_hot_lut[obs_indices]  # (B, H, W, C)
    return one_hot.permute(0, 3, 1, 2).contiguous()


def train(
    model,
    env,
    device,
    num_updates,
    epochs,
    buffer_size,
    batch_size,
    lr,
    gamma,
    lambda_,
    clip_ppo,
    c1=0.5,
    c2=0.01,
    clip_value_loss=True,
    save_interval=100,
    save_dir="./models/",
    log_dir="./logs/",
    remuse=False,
    remuse_checkpoint=None,
    remuse_logdir=None,
):
    from torch.utils.tensorboard import SummaryWriter
    current_time = time.strftime("%Y%m%d_%H%M%S")
    if remuse and remuse_logdir is not None:
        log_dir = remuse_logdir
    else:
        log_dir = os.path.join(log_dir, f"snake_ppo_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)

    obs, _ = env.reset()
    obs_tensor = _process_obs(obs, one_hot_lut=one_hot_lut, device=device)
    obs_shape = obs_tensor.shape[1:]
    num_envs = int(obs_tensor.shape[0])
    is_vector_env = hasattr(env, "num_envs")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    
    update_start = 1
    if remuse and remuse_checkpoint is not None:
        checkpoint = torch.load(remuse_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        update_start = checkpoint.get("update", 1) + 1
        print(f"Resumed model from {remuse_checkpoint} at update {update_start}")

    # Fixed rollout buffers with shape (T, N, ...)
    running_episode_rewards = np.zeros(num_envs, dtype=np.float32)

    b_actions = torch.zeros((buffer_size, num_envs), dtype=torch.int64, device=device)
    b_rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
    b_dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
    b_log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
    b_values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
    b_states = torch.zeros((buffer_size, num_envs) + obs_shape, dtype=torch.float32, device=device)

    done_count_total = 0
    for update in range(update_start, num_updates + 1):
        completed_episode_rewards = []
        completed_episode_sizes = []
        rollout_done_count = 0
        # play game
        for step in range(buffer_size):
            with torch.no_grad():
                action_logits, state_value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            if is_vector_env:
                step_action = action.detach().cpu().numpy()
            else:
                step_action = int(action.item())

            next_obs, reward, terminated, truncated, info = env.step(step_action)
            next_obs_tensor = _process_obs(next_obs, device=device, one_hot_lut=one_hot_lut)
            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            snake_size = np.asarray(info["snake_size"], dtype=np.int32)

            # Add dim for env_num_dim
            if reward.ndim == 0:
                reward = reward[None]
            if terminated.ndim == 0:
                terminated = terminated[None]
            if truncated.ndim == 0:
                truncated = truncated[None]
            if snake_size.ndim == 0:
                snake_size = snake_size[None]

            done = np.logical_or(terminated, truncated)
            b_states[step] = obs_tensor
            b_actions[step] = action
            b_rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            b_dones[step] = torch.as_tensor(done, dtype=torch.float32, device=device)
            b_log_probs[step] = log_prob
            b_values[step] = state_value.squeeze(-1)

            obs_tensor = next_obs_tensor
            
            running_episode_rewards += reward

            # Record metrics only when an env finishes an episode.
            done_indices = np.flatnonzero(done)
            if done_indices.size > 0:
                completed_episode_rewards.extend(
                    running_episode_rewards[done_indices].tolist()
                )
                completed_episode_sizes.extend(
                    snake_size[done_indices].astype(np.float32).tolist()
                )
                running_episode_rewards[done_indices] = 0.0
                rollout_done_count += int(done_indices.size)

        done_count_total += rollout_done_count
        reward_mean = (
            float(np.mean(completed_episode_rewards))
            if completed_episode_rewards
            else 0.0
        )
        reward_max = (
            float(np.max(completed_episode_rewards))
            if completed_episode_rewards
            else 0.0
        )
        snake_size_mean = (
            float(np.mean(completed_episode_sizes)) if completed_episode_sizes else 0.0
        )

        # --train--
        # Compute advantages
        b_advantages = torch.zeros((buffer_size, num_envs), device=device)
        with torch.no_grad():
            _, last_value = model(obs_tensor)
            last_value = last_value.squeeze(-1)

        last_adv = torch.zeros(num_envs, device=device)
        for t in reversed(range(buffer_size)):
            next_value = b_values[t + 1] if t != buffer_size - 1 else last_value
            non_terminal = 1.0 - b_dones[t]
            
            delta = b_rewards[t] + gamma * next_value * non_terminal - b_values[t]
            last_adv = delta + gamma * lambda_ * non_terminal * last_adv
            b_advantages[t] = last_adv
        
        b_returns = b_advantages + b_values

        # Flatten (T, N, ...) -> (T*N, ...)
        flat_states = b_states.reshape(buffer_size * num_envs, *obs_shape)
        flat_actions = b_actions.reshape(buffer_size * num_envs)
        flat_old_log_probs = b_log_probs.reshape(buffer_size * num_envs)
        flat_old_values = b_values.reshape(buffer_size * num_envs)
        flat_returns = b_returns.reshape(buffer_size * num_envs)
        flat_advantages = b_advantages.reshape(buffer_size * num_envs)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # Update epochs times
        total_batch = buffer_size * num_envs
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        total_loss_sum = 0.0
        num_minibatches = 0

        for epoch in range(epochs):
            indices = torch.randperm(total_batch, device=device)
            for start in range(0, total_batch, batch_size):
                mb_indices = indices[start : start + batch_size]

                mb_states = flat_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_old_values = flat_old_values[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]

                new_action_logits, current_values = model(mb_states)
                dist = torch.distributions.Categorical(logits=new_action_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()

                # Policy Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_ppo, 1 + clip_ppo) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                current_values = current_values.squeeze(-1)
                if clip_value_loss:
                    value_pred_clipped = mb_old_values + torch.clamp(current_values - mb_old_values,-clip_ppo,clip_ppo,)
                    value_loss_unclipped = (current_values - mb_returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(current_values, mb_returns)

                # Total Loss
                loss = policy_loss + c1 * value_loss - c2 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())
                entropy_sum += float(entropy.item())
                total_loss_sum += float(loss.item())
                num_minibatches += 1

        policy_loss_mean = policy_loss_sum / max(num_minibatches, 1)
        value_loss_mean = value_loss_sum / max(num_minibatches, 1)
        entropy_mean = entropy_sum / max(num_minibatches, 1)
        total_loss_mean = total_loss_sum / max(num_minibatches, 1)

        returns_detached = flat_returns.detach()
        values_detached = flat_old_values.detach()
        returns_var = torch.var(returns_detached)
        if float(returns_var.item()) > 1e-8:
            explained_variance = 1.0 - torch.var(returns_detached - values_detached) / (returns_var + 1e-8)
            explained_variance = float(explained_variance.item())
        else:
            explained_variance = 0.0

        writer.add_scalar("loss/policy", policy_loss_mean, update)
        writer.add_scalar("loss/value", value_loss_mean, update)
        writer.add_scalar("loss/entropy", entropy_mean, update)
        writer.add_scalar("loss/total", total_loss_mean, update)
        writer.add_scalar("rollout/reward_mean", reward_mean, update)
        writer.add_scalar("rollout/reward_max", reward_max, update)
        writer.add_scalar("rollout/snake_size_mean", snake_size_mean, update)
        writer.add_scalar("rollout/done_count", rollout_done_count, update)
        writer.add_scalar("diagnostics/explained_variance", explained_variance, update)

        print(
            f"--------------------------------------------------\n",
            f"Update {update:4d}/{num_updates} | \n",
            f"reward(mean/max) {reward_mean:7.2f}/{reward_max:7.2f} | \n",
            f"snake_size(mean) {snake_size_mean:6.2f} | \n",
            f"done {rollout_done_count:4d} | done_total {done_count_total:6d} | \n",
            f"loss(pi/v) {policy_loss_mean:8.4f}/{value_loss_mean:8.4f} | ",
            f"ev {explained_variance:7.4f} \n",
            f"--------------------------------------------------",
        )

        if update % save_interval == 0:
            save_path = os.path.join(save_dir, f"snake_ppo_ep{update}.pth")
            torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "update": update
                    }, save_path)


if __name__ == "__main__":
    env = gym.make_vec("snake-v0", num_envs=32, vectorization_mode="async")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SnakePPO(channel=4).to(device)

    train(
        model=model,
        env=env,
        device=device,
        num_updates=3000,
        epochs=4,
        buffer_size=256,
        batch_size=128,
        gamma=0.96,
        lambda_=0.95,
        lr=1e-4,
        clip_ppo=0.2,
        c1=0.3,
        c2=0.02,
        clip_value_loss=True,
        save_interval=100,
        save_dir="./models/",
        log_dir="./logs/",
        remuse=False,
        remuse_checkpoint="models/snake_ppo_ep2000.pth",
        remuse_logdir="./logs/snake_ppo_20260425_142233",
    )
    env.close()
