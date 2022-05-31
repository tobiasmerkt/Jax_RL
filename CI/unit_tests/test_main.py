import sys
import gym
from jax import numpy as jnp
import numpy as np
import pytest
sys.path.append('/tikhome/tmerkt/PycharmProjects/Jax_RL')

from Total_Background.main import *

# Create the environment
env = gym.make("CartPole-v0")

# Set seed
seed = 42
env.reset(seed=seed)
key = jax.random.PRNGKey(seed)
key1, key2, key3 = jax.random.split(key, num=3)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class TestMain:
    """
    Test suite for the main file.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        # Instantiate model
        cls.initial_state = jnp.array(env.reset(), dtype=jnp.float32)
        cls.min_episodes_criterion = 100
        cls.max_episodes = 10000
        cls.max_steps_per_episode = 1000
        cls.reward_threshold = 195
        cls.running_reward = 0
        cls.gamma = 0.99

    def test_run_episode(self):
        """
        T.
        """
        test_probs, test_values, test_rewards = run_episode(
            self.initial_state,
            model,
            actor_params,
            critic_params,
            self.max_steps_per_episode)
        assert jax.device_get(jnp.sum(test_rewards)) == jnp.shape(test_rewards)[0]
        assert 2 == 3