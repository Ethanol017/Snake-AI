import os
import time
import gymnasium as gym
import numpy as np
import torch
from model import SnakePPO, SnakeDQN
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


def train_PPO(
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
    lr_final=None,
    lr_anneal_start_update=1,
    lr_anneal_end_update=None,
    c1=0.5,
    c2=0.01,
    c2_final=None,
    c2_anneal_updates=None,
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

    # Learning-rate linear annealing config.
    if lr_final is None:
        lr_final = lr
    if lr_anneal_end_update is None:
        lr_anneal_end_update = num_updates

    # Entropy coefficient linear annealing config.
    if c2_final is None:
        c2_final = c2
    if c2_anneal_updates is None:
        c2_anneal_updates = num_updates

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

    def _linear_anneal(start_value, end_value, step, start_step, end_step):
        if end_step <= start_step:
            return end_value
        if step <= start_step:
            return start_value
        if step >= end_step:
            return end_value
        progress = (step - start_step) / (end_step - start_step)
        return start_value + (end_value - start_value) * progress

    done_count_total = 0
    for update in range(update_start, num_updates + 1):
        current_lr = _linear_anneal(
            lr,
            lr_final,
            update,
            lr_anneal_start_update,
            lr_anneal_end_update,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        anneal_progress = min(update, c2_anneal_updates) / max(c2_anneal_updates, 1)
        current_c2 = c2 + (c2_final - c2) * anneal_progress

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
                completed_episode_rewards.extend(running_episode_rewards[done_indices].tolist())
                completed_episode_sizes.extend(snake_size[done_indices].astype(np.float32).tolist())
                running_episode_rewards[done_indices] = 0.0
                rollout_done_count += int(done_indices.size)

        done_count_total += rollout_done_count
        reward_mean = float(np.mean(completed_episode_rewards)) if completed_episode_rewards else 0.0
        reward_max = float(np.max(completed_episode_rewards)) if completed_episode_rewards else 0.0
        snake_size_mean = float(np.mean(completed_episode_sizes)) if completed_episode_sizes else 0.0

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

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

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
                    value_pred_clipped = mb_old_values + torch.clamp(
                        current_values - mb_old_values,
                        -clip_ppo,
                        clip_ppo,
                    )
                    value_loss_unclipped = (current_values - mb_returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(current_values, mb_returns)

                # Total Loss
                loss = policy_loss + c1 * value_loss - current_c2 * entropy

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
        writer.add_scalar("schedule/c2", current_c2, update)
        writer.add_scalar("schedule/lr", current_lr, update)

        print(
            f"--------------------------------------------------\n",
            f"Update {update:4d}/{num_updates} | \n",
            f"reward(mean/max) {reward_mean:7.2f}/{reward_max:7.2f} | \n",
            f"snake_size(mean) {snake_size_mean:6.2f} | \n",
            f"done {rollout_done_count:4d} | done_total {done_count_total:6d} | \n",
            f"lr {current_lr:10.6g} | c2 {current_c2:7.5f} | ",
            f"loss(pi/v) {policy_loss_mean:8.4f}/{value_loss_mean:8.4f} | ",
            f"ev {explained_variance:7.4f} \n",
            f"--------------------------------------------------",
        )

        if update % save_interval == 0:
            save_path = os.path.join(save_dir, f"snake_ppo_ep{update}.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": update,
                },
                save_path,
            )


# for DQN
class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.states = np.zeros((self.capacity,) + self.obs_shape, dtype=np.uint8)
        self.next_states = np.zeros((self.capacity,) + self.obs_shape, dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add_batch(self, states, actions, rewards, next_states, dones):
        states = np.asarray(states, dtype=np.uint8)
        next_states = np.asarray(next_states, dtype=np.uint8)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        batch_size = int(states.shape[0])
        if batch_size == 0:
            return

        end = self.position + batch_size
        if end <= self.capacity:
            slice_ = slice(self.position, end)
            self.states[slice_] = states
            self.actions[slice_] = actions
            self.rewards[slice_] = rewards
            self.next_states[slice_] = next_states
            self.dones[slice_] = dones
        else:
            first = self.capacity - self.position
            second = batch_size - first
            self.states[self.position :] = states[:first]
            self.actions[self.position :] = actions[:first]
            self.rewards[self.position :] = rewards[:first]
            self.next_states[self.position :] = next_states[:first]
            self.dones[self.position :] = dones[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self.position = end % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if self.size <= batch_size:
            indices = np.random.choice(self.size, size=batch_size, replace=True)
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


def train_DQN(
    env,
    device,
    total_steps=1_000_000,
    buffer_size=100_000,
    batch_size=256,
    lr=1e-4,
    lr_final=1e-5,
    lr_anneal_start_step=0,
    lr_anneal_end_step=None,
    gamma=0.99,
    reward_divide=1.0,
    learning_starts=10_000,
    target_update_interval=5_000,
    soft_update_tau=0.005,
    save_interval=50_000,
    log_interval=1_000,
    epsilon_start=1.0,
    epsilon_final=0.05,
    epsilon_decay_steps=250_000,
    max_grad_norm=10.0,
    save_dir="./models/",
    log_root_dir="./logs/",
    resume=False,
    resume_checkpoint=None,
    resume_logdir=None,
):
    def _linear_anneal(start_value, end_value, step, start_step, end_step):
        if end_step <= start_step:
            return end_value
        if step <= start_step:
            return start_value
        if step >= end_step:
            return end_value
        progress = (step - start_step) / (end_step - start_step)
        return start_value + (end_value - start_value) * progress

    def _ensure_batch(array, dtype=None):
        value = np.asarray(array, dtype=dtype)
        if value.ndim == 0:
            value = value[None]
        return value

    def _save_checkpoint(path, step, epsilon, target_model, optimizer):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "target_model_state_dict": target_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "epsilon": epsilon,
            },
            path,
        )

    def _soft_update(target_model, source_model, tau):
        if tau <= 0.0:
            return
        with torch.no_grad():
            for target_param, source_param in zip(
                target_model.parameters(), source_model.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(source_param.data, alpha=tau)

    if lr_final is None:
        lr_final = lr
    if lr_anneal_end_step is None:
        lr_anneal_end_step = total_steps

    current_time = time.strftime("%Y%m%d_%H%M%S")
    model = SnakeDQN(channel=4).to(device)
    target_model = SnakeDQN(channel=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    log_dir = os.path.join(log_root_dir, f"snake_dqn_{current_time}")

    from torch.utils.tensorboard import SummaryWriter

    if resume and resume_logdir is not None:
        log_dir = resume_logdir
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    obs, _ = env.reset()
    obs = _ensure_batch(obs, dtype=np.uint8)
    one_hot_lut = torch.eye(4, dtype=torch.float32, device=device)
    obs_tensor = _process_obs(obs, one_hot_lut=one_hot_lut, device=device)
    obs_shape = obs.shape[1:]
    num_envs = int(obs.shape[0])
    is_vector_env = hasattr(env, "num_envs")
    action_space = getattr(env, "single_action_space", env.action_space)
    num_actions = int(action_space.n)

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    replay_buffer = ReplayBuffer(buffer_size, obs_shape)
    running_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_episode_rewards = []
    completed_episode_sizes = []
    q_loss_sum = 0.0
    q_value_sum = 0.0
    target_q_sum = 0.0
    td_error_abs_sum = 0.0
    grad_norm_sum = 0.0
    num_updates_since_log = 0
    done_count_since_log = 0
    episode_count_since_log = 0
    global_step = 0
    last_target_update_step = 0
    last_save_step = 0
    last_log_step = 0
    epsilon = epsilon_start

    if resume and resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "target_model_state_dict" in checkpoint:
            target_model.load_state_dict(checkpoint["target_model_state_dict"])
        else:
            target_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("step", 0))
        epsilon = float(checkpoint.get("epsilon", epsilon_start))
        last_target_update_step = global_step
        last_save_step = global_step
        last_log_step = global_step
        print(f"Resumed DQN model from {resume_checkpoint} at step {global_step}")

    try:
        while global_step < total_steps:
            current_lr = _linear_anneal(
                lr,
                lr_final,
                global_step,
                lr_anneal_start_step,
                lr_anneal_end_step,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            epsilon = _linear_anneal(
                epsilon_start,
                epsilon_final,
                global_step,
                0,
                epsilon_decay_steps,
            )

            with torch.no_grad():
                q_values = model(obs_tensor)
                greedy_actions = torch.argmax(q_values, dim=1)
                random_actions = torch.randint(
                    0,
                    num_actions,
                    size=(num_envs,),
                    device=device,
                )
                explore_mask = torch.rand(num_envs, device=device) < epsilon
                actions = torch.where(explore_mask, random_actions, greedy_actions)

            step_actions = actions.detach().cpu().numpy()
            if not is_vector_env and num_envs == 1:
                step_actions = int(step_actions.item())

            next_obs, reward, terminated, truncated, info = env.step(step_actions)
            next_obs = _ensure_batch(next_obs, dtype=np.uint8)
            reward = _ensure_batch(reward, dtype=np.float32)
            terminated = _ensure_batch(terminated, dtype=bool)
            truncated = _ensure_batch(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            snake_size = _ensure_batch(info["snake_size"], dtype=np.int32)
            snake_size_mask = np.asarray(
                info.get("_snake_size", np.ones_like(snake_size, dtype=bool)),
                dtype=bool,
            )
            if snake_size_mask.ndim == 0:
                snake_size_mask = snake_size_mask[None]

            clipped_reward = reward / reward_divide
            replay_buffer.add_batch(obs, step_actions, clipped_reward, next_obs, done)

            running_episode_rewards += reward

            done_indices = np.flatnonzero(done & snake_size_mask)
            if done_indices.size > 0:
                completed_episode_rewards.extend(running_episode_rewards[done_indices].tolist())
                completed_episode_sizes.extend(snake_size[done_indices].astype(np.float32).tolist())
                running_episode_rewards[done_indices] = 0.0
                done_count_since_log += int(done_indices.size)
                episode_count_since_log += int(done_indices.size)

            obs = next_obs
            obs_tensor = _process_obs(obs, one_hot_lut=one_hot_lut, device=device)
            global_step += num_envs if is_vector_env else 1

            if len(replay_buffer) >= max(learning_starts, batch_size):
                model.train()
                states, actions_batch, rewards_batch, next_states, dones_batch = (
                    replay_buffer.sample(batch_size)
                )

                states_tensor = _process_obs(states, one_hot_lut=one_hot_lut, device=device)
                next_states_tensor = _process_obs(
                    next_states, one_hot_lut=one_hot_lut, device=device
                )
                actions_tensor = torch.as_tensor(actions_batch, dtype=torch.long, device=device)
                rewards_tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=device)
                dones_tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=device)

                q_pred = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_greedy_actions = torch.argmax(
                        model(next_states_tensor), dim=1, keepdim=True
                    )
                    next_q = (
                        target_model(next_states_tensor).gather(1, next_greedy_actions).squeeze(1)
                    )
                    q_target = rewards_tensor + gamma * (1.0 - dones_tensor) * next_q

                loss = F.smooth_l1_loss(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                )
                optimizer.step()

                _soft_update(target_model, model, soft_update_tau)

                td_error = q_pred - q_target
                q_loss_sum += float(loss.item())
                q_value_sum += float(q_pred.mean().item())
                target_q_sum += float(q_target.mean().item())
                td_error_abs_sum += float(td_error.abs().mean().item())
                grad_norm_sum += grad_norm
                num_updates_since_log += 1

            if (
                target_update_interval > 0
                and global_step - last_target_update_step >= target_update_interval
            ):
                target_model.load_state_dict(model.state_dict())
                last_target_update_step = global_step

            if log_interval > 0 and global_step - last_log_step >= log_interval:
                q_loss_mean = q_loss_sum / max(num_updates_since_log, 1)
                q_value_mean = q_value_sum / max(num_updates_since_log, 1)
                target_q_mean = target_q_sum / max(num_updates_since_log, 1)
                td_error_abs_mean = td_error_abs_sum / max(num_updates_since_log, 1)
                grad_norm_mean = grad_norm_sum / max(num_updates_since_log, 1)
                reward_mean = (
                    float(np.mean(completed_episode_rewards)) if completed_episode_rewards else 0.0
                )
                reward_max = (
                    float(np.max(completed_episode_rewards)) if completed_episode_rewards else 0.0
                )
                snake_size_mean = (
                    float(np.mean(completed_episode_sizes)) if completed_episode_sizes else 0.0
                )

                writer.add_scalar("loss/q", q_loss_mean, global_step)
                writer.add_scalar("diagnostics/q_mean", q_value_mean, global_step)
                writer.add_scalar("diagnostics/target_q_mean", target_q_mean, global_step)
                writer.add_scalar("diagnostics/td_error_abs_mean", td_error_abs_mean, global_step)
                writer.add_scalar("diagnostics/grad_norm", grad_norm_mean, global_step)
                writer.add_scalar("rollout/reward_mean", reward_mean, global_step)
                writer.add_scalar("rollout/reward_max", reward_max, global_step)
                writer.add_scalar("rollout/snake_size_mean", snake_size_mean, global_step)
                writer.add_scalar("rollout/done_count", done_count_since_log, global_step)
                writer.add_scalar("rollout/episode_count", episode_count_since_log, global_step)
                writer.add_scalar("schedule/epsilon", epsilon, global_step)
                writer.add_scalar("schedule/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("replay/size", len(replay_buffer), global_step)

                print(
                    f"--------------------------------------------------\n"
                    f"Step {global_step:8d}/{total_steps} | epsilon {epsilon:6.4f} | buffer {len(replay_buffer):6d} | \n"
                    f"reward(mean/max) {reward_mean:7.2f}/{reward_max:7.2f} | snake_size(mean) {snake_size_mean:6.2f} | \n"
                    f"done {done_count_since_log:4d} | episodes {episode_count_since_log:4d} | \n"
                    f"loss(q) {q_loss_mean:8.4f} | q/target {q_value_mean:8.4f}/{target_q_mean:8.4f} | \n"
                    f"td|err| {td_error_abs_mean:7.4f} | grad_norm {grad_norm_mean:7.4f} | lr {optimizer.param_groups[0]['lr']:10.6g} \n"
                    f"--------------------------------------------------"
                )

                q_loss_sum = 0.0
                q_value_sum = 0.0
                target_q_sum = 0.0
                td_error_abs_sum = 0.0
                grad_norm_sum = 0.0
                num_updates_since_log = 0
                done_count_since_log = 0
                episode_count_since_log = 0
                completed_episode_rewards.clear()
                completed_episode_sizes.clear()
                last_log_step = global_step

            if save_interval > 0 and global_step - last_save_step >= save_interval:
                save_path = os.path.join(save_dir, f"snake_dqn_step{global_step}.pth")
                _save_checkpoint(save_path, global_step, epsilon, target_model, optimizer)
                last_save_step = global_step

        if save_interval > 0 and global_step != last_save_step:
            save_path = os.path.join(save_dir, f"snake_dqn_step{global_step}.pth")
            _save_checkpoint(save_path, global_step, epsilon, target_model, optimizer)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    training_algo = "dqn"  # "ppo" or "dqn"
    env = gym.make_vec("snake-v0", num_envs=32, vectorization_mode="sync")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if training_algo == "dqn":
        train_DQN(
            env=env,
            device=device,
            total_steps=60_000_000,
            buffer_size=500_000,
            batch_size=256,
            lr=1e-4,
            lr_final=1e-5,
            gamma=0.99,
            learning_starts=10_000,
            target_update_interval=0,
            soft_update_tau=0.005,
            save_interval=50_000,
            log_interval=1_000,
            epsilon_start=1.0,
            epsilon_final=0.01,
            epsilon_decay_steps=1_500_000,
            max_grad_norm=10.0,
            save_dir="./models/",
            log_root_dir="./logs/",
            reward_divide=15.0,
            resume=False,
            resume_checkpoint="models/snake_dqn_step20000000.pth",
            resume_logdir="logs/snake_dqn_20260507_215257",
        )
    else:
        model = SnakePPO(channel=4).to(device)

        train_PPO(
            model=model,
            env=env,
            device=device,
            num_updates=16000,
            epochs=4,
            buffer_size=512,
            batch_size=128,
            lr=1e-5,
            lr_final=1e-6,
            lr_anneal_start_update=1000,
            lr_anneal_end_update=6000,
            gamma=0.99,
            lambda_=0.95,
            clip_ppo=0.2,
            c1=0.5,
            c2=0.02,
            c2_final=0.02,
            c2_anneal_updates=3000,
            clip_value_loss=True,
            save_interval=100,
            save_dir="./models/",
            log_dir="./logs/",
            remuse=True,
            remuse_checkpoint="models/snake_ppo_ep12000.pth",
            remuse_logdir="logs/snake_ppo_20260502_160821",
        )
        env.close()
