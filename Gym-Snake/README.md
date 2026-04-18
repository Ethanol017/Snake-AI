# gym-snake

Minimal and high-throughput Snake environment for Gymnasium.

## What changed
- `snake-v0` is now a single-snake environment optimized for training throughput.
- Multi-snake gameplay and `snake-plural-v0` were removed.
- Observations are compact grid states instead of RGB pixel maps.
- Vectorized execution is supported through Gymnasium vector env APIs.

## Dependencies
- `numpy`
- `gymnasium>=0.29`
- Optional render support: `matplotlib` via `pip install -e .[render]`

## Installation
1. Clone this repository.
2. Go to the `Gym-Snake` directory.
3. Install editable package:

```bash
pip install -e .
```

## Environment API
Create env:

```python
import gymnasium as gym
import gym_snake  # registers snake-v0

env = gym.make("snake-v0")
```

Defaults:
- `grid_size=(12, 12)`
- `snake_size=3`
- `step_limit=250`
- `random_init=True`
- `render_mode=None`

### Spaces
- `action_space = Discrete(4)`
  - `0=up`, `1=right`, `2=down`, `3=left`
- `observation_space = Box(low=0, high=3, shape=(H, W), dtype=uint8)`

Observation codes:
- `0` empty
- `1` snake body
- `2` snake head
- `3` food

### Step contract
- `reset(seed=...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `info` always includes `snake_size`

Reward shape:
- Positive reward when food is eaten.
- Negative reward on death or timeout.
- Small distance-based shaping when moving.

## Vectorized usage

```python
import gymnasium as gym
import gym_snake

envs = gym.make_vec("snake-v0", num_envs=8, vectorization_mode="sync")
obs, infos = envs.reset(seed=0)

for _ in range(1000):
    actions = envs.action_space.sample()
    obs, rewards, terminated, truncated, infos = envs.step(actions)

envs.close()
```

Use `vectorization_mode="async"` for process-based parallelism.

## Rendering
- `render_mode="human"` for matplotlib debug window.
- `render_mode="rgb_array"` for frame arrays.
- Keep `render_mode=None` for maximum training speed.


