import jax
import jax.numpy as jnp
import numpy as np

RECTANGLE = jnp.zeros(1)
CUBOID = jnp.ones(1)
SPHERE = jnp.ones(1) * 2

# 假设你有一个jax数组
jax_array = jnp.array([[2.779923, 2.4769125],
                       [2.5154207, 2.6389537],
                       [2.2086883, 2.1382694],
                       [2.4731905, 1.9762284]])

print(type(jax_array))
print(jax_array.tolist())