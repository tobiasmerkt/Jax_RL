import jax.numpy as jnp


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
        self.direction = direction/jnp.linalg.norm(direction)

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
