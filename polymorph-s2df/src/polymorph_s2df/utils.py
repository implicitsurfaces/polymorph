import jax.numpy as jnp
import jax

import textwrap


def soft_plus(value):
    return jax.nn.softplus(50 * value) / 50


def soft_minus(value):
    return -jax.nn.softplus(50 * -value) / 50


def indent_shape(shape):
    return textwrap.indent(repr(shape), "  ")
