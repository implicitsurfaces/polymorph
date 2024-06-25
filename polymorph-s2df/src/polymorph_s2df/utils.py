import textwrap

import jax
import jax.numpy as jnp


def soft_plus(value):
    return jax.nn.softplus(50 * value) / 50


def soft_minus(value):
    return -jax.nn.softplus(50 * -value) / 50


def indent_shape(shape):
    return textwrap.indent(repr(shape), "  ")


def repr_point(p):
    return f"p({p[0]}, {p[1]})"


def length(coordinates):
    return jnp.linalg.norm(coordinates, axis=1)


def clamp_mask(x, low=0.0, high=1.0, softness=1e-6):
    """A smooth implementation of a mask between the boundary values"""
    lower_transition = 0.5 * (1 + jnp.tanh((x - low) / softness))
    upper_transition = 0.5 * (1 - jnp.tanh((x - high) / softness))
    return lower_transition * upper_transition
