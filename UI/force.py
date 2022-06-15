import jax.numpy as jnp
from jax import grad
from jax_md import energy


class Force:
    def __init__(self,
                 strength,
                 metric,
                 energy_model=energy.lennard_jones
                 ):
        self.strength = strength
        self.energy_model = energy_model
        self.metric = metric

    def _particle_particle_energy(self, positions: jnp.array):
        """
        Function to compute to particle-particle energy.

        Parameters
        ----------
        positions : jnp.array with (num_particles, dimension)
                array of all particle positions of dimension

        Returns
        -------
        system_energy : float
                the total energy of the system
        """
        dr = self.metric(positions, positions)
        system_energy = 1/ 2 * jnp.sum(self.energy_model(dr))
        return system_energy

    def part_part_force(self, positions: jnp.array):
        """
        Function to compute the particle-particle force on a particle. Basically just
        the negative gradient of the system energy with respect to the particle
        positions.

        Parameters
        ----------
        positions : jnp.array of (num_particles, dim)
                positions of all particles.

        Returns
        -------
        particle_particle_force : jnp.array with (1, dimension)
                force acting on the [index]-th particle caused by particle-particle
                interaction.
        """
        pa_pa_force = grad(lambda pos: -self._particle_particle_energy(pos))
        return pa_pa_force(positions)

    def __call__(self, positions, directions, actions):
        """
        A callable force function to give to the simulation engine. It combines the
        particle-particle interaction as well as the action-forces
        Parameters
        ----------
        positions : jnd.array with (particle_number, dimension)
                array with the position of all particles
        directions : jnp.array with (particle_number, dimension)
                array with the direction, the [index]-th particle is pointing at. The
                action-force will be computed along this direction.
        actionss : jnp array of int with either 1 or 0 with (particle_number,)
                translated action with 1 if the action is "step forward" (action = 0)
                or 0 else.
        Returns
        -------
        force : jnp.array with (1, dimension)
                Sum of the action-force and the particle-particle-interaction.
        """
        action_force = self.strength * jnp.where(actions == 0, 1, 0)
        return (directions.T * action_force).T + self.part_part_force(positions)
