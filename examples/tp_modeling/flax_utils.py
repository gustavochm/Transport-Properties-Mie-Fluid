from typing import Any
import jax

Array = Any

@jax.jit
def linear_activation(x: Array) -> Array:
    return x
