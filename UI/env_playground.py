import jax.numpy as jnp
from jax_md import simulate, energy, util, quantity, dataclasses, smap, space as space
import matplotlib.pyplot as plt
import environment as env

# Create the environment
num_particles = 1
num_actions = 4

static_cast = util.static_cast
Array = util.Array
f32 = util.f32
f64 = util.f64

world = env.World(dim=2, box_size=100)
eve = env.Environment(num_particles, world, reward_value=1)

action = jnp.array([0])
for i in range(10):
    feedback = eve.step(action)
    reward = feedback[0]["reward"]
    position = feedback[0]["agent"].position
    plt.scatter(eve.swarm.positions[0][0], eve.swarm.positions[0][1], label = i)
plt.grid()
plt.legend()
plt.show()
