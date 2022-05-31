import gym
import jax
from jax import jit, numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import tqdm
import collections #used for storing last 100 rewards
from matplotlib import pyplot as plt

# Create the environment
env = gym.make("CartPole-v0")

# Set seed
seed = 42
env.seed(seed)
key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(key, num=3)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

# Create model
class MLP(nn.Module):
  """Simple MLP."""
  num_actions: int
  num_hidden_units: int

  def setup(self):
      self.dense1 = nn.Dense(features=self.num_hidden_units)
      self.dense2 = nn.Dense(features=self.num_actions)

  @nn.compact
  def __call__(self, x):
      x = self.dense1(x)
      x = nn.relu(x)
      x = self.dense2(x)
      return x

class ActorCritic():
    """Combined actor critic network"""

    def __init__(self, num_actions: int, num_hidden_units: int):
        self.actor = MLP(num_actions=num_actions, num_hidden_units=num_hidden_units)
        self.critic = MLP(num_actions=1, num_hidden_units=num_hidden_units)

    def __call__(self, inputs, actor_params, critic_params):
        return (self.actor.apply(actor_params, inputs),
                self.critic.apply(critic_params, inputs))

# Instantiate model
num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions = num_actions,num_hidden_units = num_hidden_units)
dummy_input = jnp.ones(env.reset().shape)
actor_params = model.actor.init(key1, dummy_input)
critic_params = model.critic.init(key2, dummy_input)

# Collect training data
def env_step(action: jnp.array):
  """Run a single environment step in Gym"""
  state, reward, done, _ = env.step(action)
  return (jnp.array(state, dtype = jnp.float32),
          jnp.array(reward, dtype=jnp.int32),
          jnp.array(done, dtype = jnp.int32))

def run_episode(
        initial_state: jnp.array,
        model: flax.linen.Module,
        actor_params: flax.core.frozen_dict.FrozenDict,
        critic_params: flax.core.frozen_dict.FrozenDict,
        max_steps: int):
    """
    Run a single episode to collect training data
    """

    action_probs = jnp.array([], dtype=jnp.float32)
    values = jnp.array([], dtype=jnp.float32)
    rewards = jnp.array([], dtype=jnp.float32)

    for t in range(max_steps):
        action_logits_t, value = model(
            initial_state, actor_params=actor_params, critic_params=critic_params)
        action_logits_t_softmax = nn.softmax(action_logits_t)

        # Calculate log of actor probability distribution
        action_logits_t_log = jnp.log(action_logits_t_softmax)

        # Sample action from log of actor probability distribution
        rng = jax.random.PRNGKey(np.random.randint(0, 1236534623))
        action = jax.random.categorical(rng, logits=action_logits_t_log)

        # Store probs of action chosen
        action_probs = jnp.append(action_probs, action_logits_t_softmax[action])

        # Store critic values
        values = jnp.append(values, value)

        # Apply action to environment to get next state and reward
        state, reward, done = env_step(jax.device_get(action))

        # Store reward
        rewards = jnp.append(rewards, reward)

        if done == True:
            break

    return action_probs, values, rewards

# Compute expected returns
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

# Calculate Actor-Critic-Loss
def compute_loss(
        initial_state: jnp.array,
        model: flax.linen.Module,
        actor_params: flax.core.frozen_dict.FrozenDict,
        critic_params: flax.core.frozen_dict.FrozenDict,
        max_steps_per_episode: int):
    """Computes combined actor-critic loss."""

    # Run model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state=initial_state,
        model=model,
        actor_params=actor_params,
        critic_params=critic_params,
        max_steps=max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    advantage = returns - values

    action_log_probs = jnp.log(action_probs)
    actor_loss = - jnp.mean(action_log_probs * advantage)
    critic_loss = jnp.mean(
        optax.huber_loss(predictions=values, targets=returns))

    return actor_loss + critic_loss, rewards

# Full training step
def train_step(
        initial_state: jnp.array,
        model: flax.linen.Module,
        actor_params: flax.core.frozen_dict.FrozenDict,
        critic_params: flax.core.frozen_dict.FrozenDict,
        actor_optimizer,
        critic_optimizer,
        actor_opt_state,
        critic_opt_state,
        gamma: float,
        max_steps_per_episode: int):
    """Runs a model training step."""

    # Calculating loss and grad values to update network. Also calculate rewards
    (actor_grads, critic_grads), rewards = jax.grad(
        compute_loss,
        argnums=(2, 3), has_aux=True)(
        initial_state,
        model,
        actor_params,
        critic_params,
        max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Update actor
    actor_updates, actor_opt_state = actor_optimizer.update(actor_grads,
                                                            actor_opt_state,
                                                            actor_params)
    actor_params = optax.apply_updates(actor_params, actor_updates)

    # Update critic
    critic_updates, critic_opt_state = critic_optimizer.update(critic_grads,
                                                               critic_opt_state,
                                                               critic_params)
    critic_params = optax.apply_updates(critic_params, critic_updates)

    # Calculate total episode reward
    episode_reward = int(jnp.sum(rewards))

    return (episode_reward,
            (actor_params, actor_opt_state),
            (critic_params, critic_opt_state))

# Run training loop
% % time

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


