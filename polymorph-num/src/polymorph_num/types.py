from jaxtyping import Array, Float

ObsVal = Float[Array, ""]  # A rank 0 array, e.g. `jnp.array(1.0)`.

ObsDict = dict[str, ObsVal]
