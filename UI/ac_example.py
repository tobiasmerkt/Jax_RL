import jax
from jax import jit, numpy as jnp
import numpy as np
import tqdm
import collections  # used for storing last 100 rewards
from matplotlib import pyplot as plt
import time
from Total_Background.main import *

# Run loop
min_episodes_criterion = 20
max_episodes = 20
max_steps_per_episode = 1000
lr = 1e-2

# Stop when average reward >= 195 over 100 consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep running reward of last episodes
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

# Initialise state
state = create_train_state(key1, lr)
start_time = time.time() #TODO
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = jnp.array(env.reset(), dtype=jnp.float32)
        episode_reward, state = train_step(
            initial_state,
            state,
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
end_time = time.time() #TODO
print(f'\nSolved at episode {i}: average reward: {running_reward}!')
print(f'Time Difference: ', end_time - start_time)