import jax
from jax import jit, numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import tqdm
import collections  # used for storing last 100 rewards
import time
from jax_md import simulate, energy, util, quantity, dataclasses, smap, space as space
import environment as env
import matplotlib.pyplot as plt

# Create the environment
num_particles = 1
num_actions = 4
num_hidden_units = 32

static_cast = util.static_cast
Array = util.Array
f32 = util.f32
f64 = util.f64

world = env.World(dim=2, box_size=100)
eve = env.Environment(num_particles, world, reward_value=1)

# Set seed
seed = 0
key = jax.random.PRNGKey(seed)
key1, key2 = jax.random.split(key)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()  # used when dividing values


class ActorCritic(nn.Module):
    """Combined actor critic network"""
    num_actions: int
    num_hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.num_hidden_units)(x))
        actor = nn.Dense(self.num_actions)(x)
        critic = nn.Dense(1)(x)
        return actor, critic


def create_train_state(rng, learning_rate):
    """Creates initial TrainState."""
    model = ActorCritic(
        num_actions=num_actions,
        num_hidden_units=num_hidden_units)
    params = model.init(rng, jnp.ones_like(eve.swarm.positions[0]))
    optimizer = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)


# Collect training data
def eve_step(action: jnp.array):
    feedback = eve.step(action)
    reward = feedback[0]["reward"]
    position = feedback[0]["agent"].position
    return reward, position


def run_episode(
        initial_state: jnp.array,
        params: flax.core.frozen_dict.FrozenDict,
        max_steps: int):
    """
    Run a single episode to collect training data
    """

    action_probs = jnp.array([], dtype=jnp.float32)
    values = jnp.array([], dtype=jnp.float32)
    rewards = jnp.array([], dtype=jnp.float32)

    apply_fn = ActorCritic(num_actions, num_hidden_units).apply
    apply_fn_jit = jit(apply_fn)
    softmax_fn = nn.softmax
    softmax_fn_jit = jit(softmax_fn)

    for t in range(max_steps):
        initial_state = ((initial_state - jnp.mean(initial_state)) / (jnp.std(initial_state) + eps)) #normalise
        action_logits_t, value = apply_fn_jit(params, initial_state)

        # Sample action from action probability distribution
        rng = jax.random.PRNGKey(np.random.randint(0, 1236534623))
        action = jax.random.categorical(rng, logits=action_logits_t)
        action_probs_t = softmax_fn_jit(action_logits_t)

        # Store probs of action chosen
        action_probs = jnp.append(action_probs, action_probs_t[action])

        # Store critic values
        values = jnp.append(values, value)

        # Apply action to environment to get next state and reward
        reward, position = eve_step(action)

        # Store reward
        rewards = jnp.append(rewards, reward)
        initial_state = position

    return action_probs, values, rewards


# Compute expected return
@jit
def get_expected_return(
        rewards: jnp.array,
        gamma: float,
        standardize: bool = True) -> jnp.array:
    """Compute expected returns per timestep"""

    rewards = rewards[::-1]
    discounted_sum = 0.0
    returns = jnp.array([], dtype=jnp.float32)

    for i in range(len(rewards)):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        returns = jnp.append(returns, discounted_sum)
    returns = returns[::-1]

    if standardize:
        returns = ((returns - jnp.mean(returns)) / (jnp.std(returns) + eps))

    return returns


# Shared Actor-Critic loss
def compute_loss(
        initial_state: jnp.array,
        params: flax.core.frozen_dict.FrozenDict,
        max_steps_per_episode: int):
    """Computes combined actor-critic loss."""

    # Run model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state=initial_state,
        params=params,
        max_steps=max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)
    advantage = returns - values

    action_log_probs = jnp.log(action_probs)
    actor_loss = - jnp.sum(action_log_probs * advantage)

    critic_loss = jnp.sum(
        optax.huber_loss(predictions=values, targets=returns))

    return actor_loss + critic_loss, rewards


# Training Step
def train_step(
        initial_state: jnp.array,
        state: flax.core.frozen_dict.FrozenDict,
        gamma: float,
        max_steps_per_episode: int):
    """Runs a model training step."""

    # Calculating loss and grad values to update network. Also calculate rewards
    grads, rewards = jax.grad(
        compute_loss,
        argnums=(1), has_aux=True)(
        initial_state,
        state.params,
        max_steps_per_episode)

    # Update state
    state = state.apply_gradients(grads=grads)

    # Calculate total episode reward
    episode_reward = int(jnp.sum(rewards))

    return episode_reward, state

# Run loop
min_episodes_criterion = 20
max_episodes = 50
max_steps_per_episode = 300
lr = 1e-1

# Stop when average reward >= 195 over 100 consecutive trials
reward_threshold = 9e5
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
        initial_state = eve.reset()
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

        if i > max_episodes:
            break

        # if running_reward > reward_threshold and i >= min_episodes_criterion:
        #     break
end_time = time.time() #TODO
print(f'\nSolved at episode {i}: average reward: {running_reward}!')
print(f'Time Difference: ', end_time - start_time)

# Visualise results
apply_fn = ActorCritic(num_actions, num_hidden_units).apply
apply_fn_jit = jit(apply_fn)

positions = jnp.array([])
rewards = jnp.array([])
for i in range(100):
    initial_state = ((initial_state - jnp.mean(initial_state)) / (
                jnp.std(initial_state) + eps))  # normalise
    action_logits_t, value = apply_fn_jit(state.params, initial_state)

    # Sample action from action probability distribution
    rng = jax.random.PRNGKey(np.random.randint(0, 1236534623))
    action = jax.random.categorical(rng, logits=action_logits_t)
    feedback = eve.step(action)
    reward = feedback[0]["reward"]
    position = feedback[0]["agent"].position
    positions = jnp.append(positions, position)
    rewards = jnp.append(rewards, reward)

jnp.save('params', state.params)
jnp.save('positions_backup', positions)
jnp.save('rewards_backup', rewards)

plt.scatter(positions[::2], positions[1::2])
plt.title('Positions')
plt.grid()
plt.show()

plt.plot(rewards)
plt.title('Rewards')
plt.grid()
plt.show()