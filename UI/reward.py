import jax.numpy as jnp


class FindLocation:
    def __init__(self,
                 size: float,
                 dim: int,
                 reward_value,
                 metric,
                 particle_number,
                 location=None):
        self.size = size
        self.dim = dim
        self.reward_value = reward_value
        self.metric = metric
        self.particle_number = particle_number
        self.location = location

    def reward(self, positions: jnp.array):
        """

        Parameters
        ----------
        positions

        Returns
        -------

        """
        if self.location is None:
            self.location = jnp.array([jnp.ones(self.dim) * 0.5])

        rescaled_positions = positions / self.size
        rewards = 2 * self.reward_value / (jnp.sqrt(2)) * (jnp.sqrt(2) / 2 - self.metric(
            rescaled_positions,
            self.location))
        if self.particle_number != 1:
            return jnp.squeeze(rewards)
        else:
            return rewards
