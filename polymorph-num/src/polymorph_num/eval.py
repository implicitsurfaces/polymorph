import jax
import jax.numpy as jnp

from . import expr as e

__all__ = ["_eval"]


def _eval(expr: e.Expr, params, param_map, obs_dict) -> jax.Array:
    match expr:
        case e.Scalar(value):
            return jnp.array(value)

        case e.Arr(value):
            return jnp.array(value)

        case e.Broadcast(orig, _dim):
            return _eval(orig, params, param_map, obs_dict)

        case e.Unary(orig, op):
            o = _eval(orig, params, param_map, obs_dict)
            match op:
                case e.UnOp.Sqrt:
                    return jnp.sqrt(o)
                case e.UnOp.Sigmoid:
                    return jax.nn.sigmoid(o)
                case e.UnOp.SmoothAbs:
                    return o * jnp.tanh(10.0 * o)
                case e.UnOp.Abs:
                    return jnp.abs(o)
                case e.UnOp.SoftPlus:
                    return jax.nn.softplus(50 * o) / 50
                case e.UnOp.Log:
                    return jnp.log(o)
                case e.UnOp.Cos:
                    return jnp.cos(o)
                case e.UnOp.Sin:
                    return jnp.sin(o)
                case e.UnOp.Sign:
                    return jnp.sign(o)
                case e.UnOp.Tanh:
                    return jnp.tanh(o)
                case e.UnOp.ArcTan:
                    return jnp.atan(o)

        case e.Binary(left, right, op):
            l = _eval(left, params, param_map, obs_dict)
            r = _eval(right, params, param_map, obs_dict)
            match op:
                case e.BinOp.Add:
                    return l + r
                case e.BinOp.Sub:
                    return l - r
                case e.BinOp.Mul:
                    return l * r
                case e.BinOp.Div:
                    return l / r
                case e.BinOp.Exp:
                    return l**r
                case e.BinOp.Min:
                    return jnp.minimum(l, r)
                case e.BinOp.Max:
                    return jnp.maximum(l, r)
                case e.BinOp.ArcTan2:
                    return jnp.arctan2(l, r)
                case e.BinOp.Mod:
                    return jnp.mod(l, r)

        case e.ComparisonIf(a, b, condition_true, condition_false, op):
            a_ = _eval(a, params, param_map, obs_dict)
            b_ = _eval(b, params, param_map, obs_dict)
            if op == e.ComparisonOp.Gt:
                return jnp.where(
                    a_ > b_,
                    _eval(condition_true, params, param_map, obs_dict),
                    _eval(condition_false, params, param_map, obs_dict),
                )
            elif op == e.ComparisonOp.Ge:
                return jnp.where(
                    a_ >= b_,
                    _eval(condition_true, params, param_map, obs_dict),
                    _eval(condition_false, params, param_map, obs_dict),
                )
            elif op == e.ComparisonOp.Eq:
                return jnp.where(
                    a_ == b_,
                    _eval(condition_true, params, param_map, obs_dict),
                    _eval(condition_false, params, param_map, obs_dict),
                )

        case e.Param():
            return param_map.get(expr, params)

        case e.Observation(name):
            return obs_dict[name]

        case e.Sum(orig):
            return jnp.sum(_eval(orig, params, param_map, obs_dict))

        case e.Debug(tag, orig):
            o = _eval(orig, params, param_map, obs_dict)
            print(tag, o)
            return o

        case _:
            raise ValueError(f"Unknown expression: {expr}")
