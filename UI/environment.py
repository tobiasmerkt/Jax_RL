import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate, energy, util, quantity, dataclasses, smap
from jax_md import space as space

static_cast = util.static_cast
Array = util.Array
f32 = util.f32
f64 = util.f64


class Agent:
    """
    Agent class to contain all the information about an agent.
    """

    def __init__(
            self,
            position: jnp.array,
            direction: jnp.array,
            index: int):
        self.position = position
        self.index = index
        self.direction = direction / jnp.linalg.norm(direction)

    def set_position(self, new_position):
        """
        A method to update the position of a particle
        Parameters
        ----------
        new_position : jnp.array with (1, dimension)

        Returns
        -------

        """
        self.position = new_position

    def set_direction(self, new_direction):
        """
        A method to update the direction of a particle
        Parameters
        ----------
        new_direction : jnp.array with (1, dimension)
                non-zero direction
        Returns
        -------

        """
        self.direction = new_direction / jnp.linalg.norm(new_direction)


class Swarm:
    """
    Swarm class to contain all the information about a group of agents.
    """

    def __init__(self, agents: list):
        self.agents = agents
        self.positions = jnp.array([agent.position for agent in agents])
        self.directions = jnp.array([agent.direction for agent in agents])
        self.size = len(self.agents)
        self.state = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.size:
            next_agent = self.agents[self.count]
            self.count += 1
        else:
            raise StopIteration
        return next_agent

    def set_directions(self, new_directions):
        self.directions = jnp.array(new_directions)
        for i in range(self.size):
            self.agents[i].set_direction(new_directions[i])

    def set_positions(self, new_positions):
        self.positions = jnp.array(new_positions)
        for i in range(self.size):
            self.agents[i].set_position(new_positions[i])

    def __call__(self, index):
        return self.agents[index]


class World:
    def __init__(self,
                 dim,
                 box_size):
        self.dim = dim
        self.size = box_size
        displacement, self.shift = space.periodic(box_size)
        metric = space.metric(displacement)
        self.displacement = space.map_product(displacement)
        self.metric = space.map_product(metric)


class Force:
    def __init__(self,
                 force,
                 metric,
                 energy_model=energy.lennard_jones
                 ):
        self.force = force
        self.energy_model = energy_model
        self.metric = metric

    def _particle_particle_energy(self, position: jnp.array):
        """
        Function to compute to particle-particle energy.

        Parameters
        ----------
        position : jnp.array with (num_particles, dimension)
                array of all particle positions of dimension

        Returns
        -------
        system_energy : float
                the total energy of the system
        """
        dr = self.metric(position, position)
        system_energy = 1 / 2 * jnp.sum(self.energy_model(dr))
        return system_energy

    def part_part_force(self, positions):
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
        pa_pa_force = jax.grad(lambda pos: -self._particle_particle_energy(pos))
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
        action_force = self.force * jnp.where(actions == 0, 1, 0)
        return (directions.T * action_force).T + self.part_part_force(positions)


class EnvState:
    def __init__(self,
                 positions,
                 directions,
                 simulation_state,
                 ):
        self.positions = positions
        self.directions = directions
        self.simulation_state = simulation_state


class Environment:
    def __init__(
            self,
            particle_number,
            world,
            dt=0.1,
            kt=0,
            force_value=2.0,
            rotation_angle=15,
            reward_value=10):

        self.particle_number = particle_number
        self.dim = world.dim
        self.size = world.size
        self.dt = dt
        self.kt = kt
        self.force_value = force_value
        self.rotation_angle = rotation_angle
        self.reward_value = reward_value

        # get the shift, displacement and metric from the used world.
        self.shift = world.shift
        self.displacement = world.displacement
        self.metric = world.metric

        # creat the "num_particles" agents randomly distributed over the box.
        key = jax.random.PRNGKey(0)
        key, split_positions = jax.random.split(key)
        key, split_directions = jax.random.split(key)
        pos = jax.random.uniform(split_positions,
                                 (self.particle_number, self.dim),
                                 maxval=self.size)

        directs = jax.random.uniform(split_directions,
                                     (self.particle_number, self.dim),
                                     maxval=self.size)
        directs = (directs.T / jnp.linalg.norm(directs, axis=1)).T

        agents = []
        for i in range(particle_number):
            agents.append(Agent(position=pos[i],
                                direction=directs[i],
                                index=i))
        self.agents = agents
        self.swarm = Swarm(agents)

        # Initialize the force and the jax-simulation.
        self.force = Force(force=self.force_value, metric=self.metric)
        init_fn, self.apply_fn = simulate.brownian(self.force,
                                                   self.shift,
                                                   self.dt,
                                                   self.kt)

        # Update the JAX simulations state of the swarm
        self.swarm.state = init_fn(jax.random.PRNGKey(0), self.swarm.positions)

    def _reward(self, positions, location=None):
        """
        Reward function to compute the reward with.

        Parameters
        ----------
        positions : jnp.array of (1, dim)
                position of the particle

        Returns
        -------
        reward : float
                reward of the particle for its state.
        """
        if location is None:
            location = jnp.array([jnp.ones(self.dim) * 0.5])

        rescaled_positions = positions / self.size
        reward = self.reward_value / jnp.sqrt(2) * (jnp.sqrt(2) - self.metric(
            rescaled_positions,
            location))
        if self.particle_number != 1:
            return jnp.squeeze(reward)
        else:
            return reward

    def step(self, actions: Array):
        """
        Function to perform one step of the environment.

        Parameters
        ----------
        actions : jnp.array ints (elements of [0 ,1 ,2, 3]) of length particle_number.
                Actions chosen by each agent.

        Returns
        -------
        reeturn : list of dictionaries

        """
        if actions.shape[0] < self.particle_number:
            raise ValueError(f"Not enough actions ({actions.shape[0]}) "
                             f"for all agents ({self.particle_number})")
        elif actions.shape[0] > self.particle_number:
            raise ValueError(f"Too many actions. More actions ({actions.shape[0]}) "
                             f"than agents ({self.particle_number})")

        def rotation(_actions):
            theta = np.radians(self.rotation_angle)
            rotate_right = jnp.array(((jnp.cos(theta), -jnp.sin(theta)),
                                      (jnp.sin(theta), jnp.cos(theta))))
            rotate_left = jnp.array(((jnp.cos(theta), jnp.sin(theta)),
                                     (-jnp.sin(theta), jnp.cos(theta))))
            rotation_operator = []
            for i in range(_actions.shape[0]):
                if actions[i] == 1:
                    rotation_operator.append(rotate_right)
                elif _actions[i] == 2:
                    rotation_operator.append(rotate_left)
                else:
                    rotation_operator.append(jnp.identity(2))
            rotation_operator = jnp.array(rotation_operator)

            new_directions = []
            for i in range(len(self.swarm.directions)):
                new_directions.append(rotation_operator[i] @ self.swarm.directions[i])
            self.swarm.set_directions(new_directions)
            return

        rotation(actions)

        self.swarm.state = self.apply_fn(self.swarm.state,
                                         directions=self.swarm.directions,
                                         actions=actions)

        self.swarm.set_positions(self.swarm.state.position)

        reward = self._reward(self.swarm.positions)

        environment_return = []
        for index, agent in enumerate(self.swarm):
            environment_return.append({"agent": agent, "reward": reward[index]})

        return environment_return

    def reset(self, seed):
        return

    def render(self):
        return

    def action_space(self):
        return print("Discrete(4)")

    def observation_space(self):
        return

    def reward_range(self):
        return

    def close(self):
        return