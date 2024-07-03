from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

ScreenPos = Tuple[float, float]


@dataclass(frozen=True)
class WorldPos:
    x: float
    y: float

    def as_array(self) -> Float[Array, "2"]:
        return jnp.array([self.x, self.y])
