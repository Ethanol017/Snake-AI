import gymnasium as gym
import numpy as np
import gym_snake  # type: ignore  # Registers snake-v0


def describe_array(name, value):
    arr = np.asarray(value)
    print(f"{name:<16} shape={arr.shape}, dtype={arr.dtype}")


def describe_infos(infos):
    print(f"infos type       {type(infos).__name__}")

    if isinstance(infos, dict):
        for key, value in infos.items():
            arr = np.asarray(value)
            print(f"infos[{key!r}]   shape={arr.shape}, dtype={arr.dtype}")
        return

    if isinstance(infos, (list, tuple)):
        print(f"infos length     {len(infos)}")
        if len(infos) > 0 and isinstance(infos[0], dict):
            keys = sorted({k for item in infos for k in item.keys()})
            for key in keys:
                values = [item.get(key, None) for item in infos]
                arr = np.asarray(values)
                print(f"infos[{key!r}]   shape={arr.shape}, dtype={arr.dtype}")
        return

    print(f"infos value      {infos}")


if __name__ == "__main__":
    num_envs = 8
    num_steps = 3

    vec_env = gym.make_vec("snake-v0", num_envs=num_envs, vectorization_mode="sync")
    # vec_env = gym.make("snake-v0")
    try:
        obs, infos = vec_env.reset(seed=0)
        print("=== reset ===")
        describe_array("obs", obs)
        describe_infos(infos)

        for step_idx in range(1, num_steps + 1):
            actions = vec_env.action_space.sample()
            obs, rewards, terminated, truncated, infos = vec_env.step(actions)

            print(f"\n=== step {step_idx} ===")
            describe_array("actions", actions)
            describe_array("obs", obs)
            describe_array("rewards", rewards)
            describe_array("terminated", terminated)
            describe_array("truncated", truncated)
            describe_infos(infos)
    finally:
        vec_env.close()
