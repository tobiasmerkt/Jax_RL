import sys
import gym
import jax
from jax import numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import tqdm
import collections #used for storing last 100 rewards
from matplotlib import pyplot as plt
sys.path.append('/tikhome/tmerkt/PycharmProjects/Jax_RL')
import Total_Background as rl

# TODO: fix run file

# Run training loop
min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Stop when average reward >= 195 over 100 consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep running reward of last 100 episodes
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

# Initialise optimiser
actor_optimizer = optax.adam(learning_rate=0.01)
actor_opt_state = actor_optimizer.init(actor_params)

critic_optimizer = optax.adam(learning_rate=0.01)
critic_opt_state = critic_optimizer.init(critic_params)

with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = jnp.array(env.reset(), dtype=jnp.float32)
        (episode_reward,
         (actor_params, actor_opt_state),
         (critic_params, critic_opt_state)) = train_step(
            initial_state,
            model,
            actor_params,
            critic_params,
            actor_optimizer,
            critic_optimizer,
            actor_opt_state,
            critic_opt_state,
            gamma,
            max_steps_per_episode)

        episodes_reward.append(jax.device_get(episode_reward))
        running_reward = np.mean(episodes_reward)

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
            print('.')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward}!')