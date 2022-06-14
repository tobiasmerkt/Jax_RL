import jax.numpy as jnp
import numpy as np
import jax
import jax_md
from jax_md import simulate, energy, util, quantity, dataclasses, smap, space as space
from jax_md import quantity
from jax_md.quantity import box_size_at_number_density

num_particles = 10
num_actions = 4

static_cast = util.static_cast
Array = util.Array
f32 = util.f32
f64 = util.f64

import environment as env
world = env.World(dim=2, box_size=100)
forcer = env.Force(force=2, metric=world.metric)
eve = env.Environment(num_particles, world)

actions = np.random.randint(0,num_actions, num_particles)
print(f'{actions=}')
print(eve.step(actions))
print(eve.swarm.positions)