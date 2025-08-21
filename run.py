import os
# Optional: speed up reloads by caching Madrona kernels (if you use the compiled backend)
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Ensure Madrona can pre-allocate before JAX

from datetime import datetime
import functools

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from flax import linen
from IPython.display import clear_output, display
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np

from mujoco_playground import manipulation
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
from mujoco_playground.config import manipulation_params

np.set_printoptions(precision=3, suppress=True, linewidth=100)

env_name = "PandaPickCubeCartesian"
env_cfg = manipulation.get_default_config(env_name)

num_envs = int(os.environ.get("NUM_ENVS", "1024"))
episode_length = int(4 / env_cfg.ctrl_dt)

# Rasterizer is less feature-complete than ray-tracing backend but stable
config_overrides = {
    "episode_length": episode_length,
    "vision": True,
    "obs_noise.brightness": [0.75, 2.0],
    "vision_config.use_rasterizer": False,
    "vision_config.render_batch_size": num_envs,
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
    "box_init_range": 0.1,  # +- 10 cm
    "action_history_length": 5,
    "success_threshold": 0.03,
}

env = manipulation.load(env_name, config=env_cfg, config_overrides=config_overrides)

randomization_fn = functools.partial(
    randomize.domain_randomize,
    num_worlds=num_envs,
)

env = wrapper.wrap_for_brax_training(
    env,
    vision=True,
    num_vision_envs=num_envs,
    episode_length=episode_length,
    action_repeat=1,
    randomization_fn=randomization_fn,
)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

def tile(img, d):
    assert img.shape[0] == d * d
    img = img.reshape((d, d) + img.shape[1:])
    return np.concatenate(np.concatenate(img, axis=1), axis=1)

def unvmap(x):
    return jax.tree_util.tree_map(lambda y: y[0], x)

state = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))

# Display a tiled sample of initial pixels (best-effort; safe to skip in headless)
try:
    media.show_image(tile(state.obs['pixels/view_0'][:64], 8), width=512)
except Exception as e:
    print("Skipping inline image display:", e)

state = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
rollout = [unvmap(state)]

f = 0.2
for i in range(env_cfg.episode_length):
    action = []
    for j in range(env.action_size):
        action.append(
            jp.sin(
                unvmap(state.data.time) * 2 * jp.pi * f + j * 2 * jp.pi / env.action_size
            )
        )
    action = jp.tile(jp.array(action), (num_envs, 1))
    state = jit_step(state, action)
    rollout.append(unvmap(state))

# Render rollout (best-effort; safe to skip in headless)
try:
    frames = env.render(rollout)
    media.show_video(frames, fps=1.0 / env.dt)
except Exception as e:
    print("Skipping inline rollout video:", e)

network_factory = functools.partial(
    ppo_networks_vision.make_ppo_networks_vision,
    policy_hidden_layer_sizes=[256, 256],
    value_hidden_layer_sizes=[256, 256],
    activation=linen.relu,
    normalise_channels=True,
)

ppo_params = manipulation_params.brax_vision_ppo_config(env_name)
ppo_params.num_timesteps = int(os.environ.get("NUM_TIMESTEPS", "7000000"))
ppo_params.num_envs = num_envs
ppo_params.num_eval_envs = num_envs
del ppo_params.network_factory
ppo_params.network_factory = network_factory

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    clear_output(wait=True)
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics.get("eval/episode_reward", 0.0))
    y_dataerr.append(metrics.get("eval/episode_reward_std", 0.0))

    steps = ppo_params["num_timesteps"]
    plt.figure()
    plt.xlim([steps * -0.1, steps * 1.25])
    plt.ylim([0, 14])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr)
    try:
        display(plt.gcf())
    except Exception:
        # In headless, just save a snapshot periodically
        outdir = "figures"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"progress_{num_steps}.png"))
    plt.close()

train_fn = functools.partial(
    ppo.train,
    augment_pixels=True,
    **dict(ppo_params),
    progress_fn=progress,
)

make_inference_fn, params, metrics = train_fn(environment=env)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(0)
rollout = []
n_episodes = 1
to_keep = 256

def keep_until(state, i):
    return jax.tree_util.tree_map(lambda x: x[:i], state)

for _ in range(n_episodes):
    key_rng = jax.random.split(rng, num_envs)
    state = jit_reset(key_rng)
    rollout.append(keep_until(state, to_keep))
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        act_rng = jax.random.split(act_rng, num_envs)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(keep_until(state, to_keep))

render_every = 1
try:
    frames = env.render([unvmap(s) for s in rollout][::render_every])
    rewards = [unvmap(s).reward for s in rollout]
    media.show_video(frames, fps=1.0 / env.dt / render_every)
except Exception as e:
    print("Skipping inline eval video:", e)

# Plot rewards (best-effort)
try:
    import matplotlib.pyplot as plt
    rewards = [np.array(unvmap(s).reward) for s in rollout]
    plt.figure(figsize=(3, 2))
    plt.plot(rewards)
    plt.xlabel("time step")
    plt.ylabel("reward")
    plt.show()
except Exception:
    pass

# Save tiled observation video (best-effort)
try:
    obs = [np.array(s.obs['pixels/view_0']) for s in rollout]
    obs = [tile(img, int(np.sqrt(to_keep))) for img in obs]
    media.show_video(obs, fps=1.0 / env_cfg.ctrl_dt, width=512)
except Exception as e:
    print("Skipping obs video:", e)

print("Done.")
