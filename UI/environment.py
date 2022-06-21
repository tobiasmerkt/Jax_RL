import jax.numpy as jnp
import numpy as np
from jax import random
from jax_md import simulate, energy, util, quantity, dataclasses, smap
from jax_md import space as space

from agent import Agent, Swarm
import reward as task
import force as force

static_cast = util.static_cast
Array = util.Array
f32 = util.f32
f64 = util.f64


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


class Environment:
    def __init__(
            self,
            particle_number,
            world: World,
            dt=0.1,
            kt=0,
            strength=2.0,
            rotation_angle=15,
            reward_value=10):

        self.particle_number = particle_number
        self.dim = world.dim
        self.size = world.size
        self.dt = dt
        self.kt = kt
        self.strength = strength
        self.rotation_angle = rotation_angle
        self.reward_value = reward_value

        # get the shift, displacement and metric from the used world.
        self.shift = world.shift
        self.displacement = world.displacement
        self.metric = world.metric

        # creat the "num_particles" agents randomly distributed over the box.
        key = random.PRNGKey(0)
        key, split_positions = random.split(key)
        self.reset_key, split_directions = random.split(key)
        pos = random.uniform(split_positions,
                             (self.particle_number, self.dim),
                             maxval=self.size)

        directs = random.uniform(split_directions,
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

        # Initialize the force
        self.force = force.Force(strength=self.strength, metric=self.metric)

        # Initialize the reward function
        reward_class = task.FindLocation(size=self.size,
                                         dim=self.dim,
                                         reward_value=1,
                                         metric=self.metric,
                                         particle_number=self.particle_number)

        self.reward = reward_class.reward

        # Initialize the JaxMD simulation state
        init_fn, self.apply_fn = simulate.brownian(self.force,
                                                   self.shift,
                                                   self.dt,
                                                   self.kt)

        # Update the JAX simulations state of the swarm
        self.swarm.state = init_fn(random.PRNGKey(0), self.swarm.positions)

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

        try:
            action_shape = actions.shape[0]
        except:
            action_shape = 1

        if action_shape < self.particle_number:
            raise ValueError(f"Not enough actions ({actions.shape[0]}) "
                             f"for all agents ({self.particle_number})")
        elif action_shape > self.particle_number:
            raise ValueError(f"Too many actions. More actions ({actions.shape[0]}) "
                             f"than agents ({self.particle_number})")

        def rotation(_actions):
            theta = jnp.radians(self.rotation_angle)
            rotate_right = jnp.array(((jnp.cos(theta), -1 * jnp.sin(theta)),
                                      (jnp.sin(theta), jnp.cos(theta))))
            rotate_left = jnp.array(((jnp.cos(theta), jnp.sin(theta)),
                                     (-1 * jnp.sin(theta), jnp.cos(theta))))
            rotation_operator = []
            for i in range(action_shape):
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

        def single_rotation(_action):
            theta = jnp.radians(self.rotation_angle)
            rotate_right = jnp.array(((jnp.cos(theta), -1 * jnp.sin(theta)),
                                      (jnp.sin(theta), jnp.cos(theta))))
            rotate_left = jnp.array(((jnp.cos(theta), jnp.sin(theta)),
                                     (-1 * jnp.sin(theta), jnp.cos(theta))))
            rotation_operator = []
            if _action == 1:
                rotation_operator.append(rotate_right)
            elif _action == 2:
                rotation_operator.append(rotate_left)
            else:
                rotation_operator.append(jnp.identity(2))
            rotation_operator = jnp.array(rotation_operator)

            new_directions = []
            for i in range(len(self.swarm.directions)):
                new_directions.append(rotation_operator[i] @ self.swarm.directions[i])
            self.swarm.set_directions(new_directions)
            return

        if self.particle_number != 1:
            rotation(actions)
        else:
            single_rotation(actions)

        self.swarm.state = self.apply_fn(self.swarm.state,
                                         directions=self.swarm.directions,
                                         actions=actions)

        self.swarm.set_positions(self.swarm.state.position)

        reward = self.reward(self.swarm.positions)

        environment_return = []
        for index, agent in enumerate(self.swarm):
            environment_return.append({"agent": agent, "reward": reward})

        return environment_return

    def reset(self):
        self.reset_key, split_positions = random.split(self.reset_key)
        self.reset_key, split_directions = random.split(self.reset_key)
        pos = random.uniform(split_positions,
                             (self.particle_number, self.dim),
                             maxval=self.size)

        """
        directs = random.uniform(split_directions,
                                 (self.particle_number, self.dim),
                                 maxval=self.size)
        directs = (directs.T / jnp.linalg.norm(directs, axis=1)).T
        """

        self.swarm.set_positions(pos)
        # self.swarm.set_directions(directs)
        return self.swarm.positions[0]

    def render(self):
        return

    def action_space(self):
        return 4

    def observation_space(self):
        return

    def reward_range(self):
        return

    def close(self):
        return