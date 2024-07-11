import jax
import jax.numpy as jnp

from . import expr as e

__all__ = ["_eval"]


def _eval(expr: e.Expr, params, param_map, obs_dict, memo) -> jax.Array:
    if id(expr) in memo:
        return memo[id(expr)]
    else:
        result = None
        match expr:
            case e.Scalar(value):
                result = jnp.array(value)

            case e.Arr(value):
                result = jnp.array(value)

            case e.Broadcast(orig, _dim):
                result = _eval(orig, params, param_map, obs_dict, memo)

            case e.Unary(orig, op, constants):
                o = _eval(orig, params, param_map, obs_dict, memo)
                match op:
                    case e.UnOp.Sqrt:
                        result = jnp.sqrt(o)
                    case e.UnOp.Sigmoid:
                        result = jax.nn.sigmoid(o)
                    case e.UnOp.SmoothAbs:
                        result = o * jnp.tanh(10.0 * o)
                    case e.UnOp.Abs:
                        result = jnp.abs(o)
                    case e.UnOp.SoftPlus:
                        result = jax.nn.softplus(50 * o) / 50
                    case e.UnOp.Log:
                        result = jnp.log(o)
                    case e.UnOp.Cos:
                        result = jnp.cos(o)
                    case e.UnOp.Sin:
                        result = jnp.sin(o)
                    case e.UnOp.Sign:
                        result = jnp.sign(o)
                    case e.UnOp.Tanh:
                        result = jnp.tanh(o)
                    case e.UnOp.ArcTan:
                        result = jnp.atan(o)
                    case e.UnOp.Boxcar:
                        min, max = constants
                        return jnp.where((o >= min) & (o <= max), 1.0, 0.0)

            case e.Binary(left, right, op):
                l = _eval(left, params, param_map, obs_dict, memo)
                r = _eval(right, params, param_map, obs_dict, memo)
                match op:
                    case e.BinOp.Add:
                        result = l + r
                    case e.BinOp.Sub:
                        result = l - r
                    case e.BinOp.Mul:
                        result = l * r
                    case e.BinOp.Div:
                        result = l / r
                    case e.BinOp.Exp:
                        result = l**r
                    case e.BinOp.Min:
                        result = jnp.minimum(l, r)
                    case e.BinOp.Max:
                        result = jnp.maximum(l, r)
                    case e.BinOp.ArcTan2:
                        result = jnp.arctan2(l, r)
                    case e.BinOp.Mod:
                        result = jnp.mod(l, r)

            case e.ComparisonIf(a, b, condition_true, condition_false, op):
                a_ = _eval(a, params, param_map, obs_dict)
                b_ = _eval(b, params, param_map, obs_dict)
                if op == e.ComparisonOp.Gt:
                    result = jnp.where(
                        a_ > b_,
                        _eval(condition_true, params, param_map, obs_dict, memo),
                        _eval(condition_false, params, param_map, obs_dict, memo),
                    )
                elif op == e.ComparisonOp.Ge:
                    result = jnp.where(
                        a_ >= b_,
                        _eval(condition_true, params, param_map, obs_dict, memo),
                        _eval(condition_false, params, param_map, obs_dict, memo),
                    )
                elif op == e.ComparisonOp.Eq:
                    result = jnp.where(
                        a_ == b_,
                        _eval(condition_true, params, param_map, obs_dict, memo),
                        _eval(condition_false, params, param_map, obs_dict, memo),
                    )

            case e.Param():
                result = param_map.get(expr, params)

            case e.Observation(name):
                result = obs_dict[name]

            case e.Sum(orig):
                result = jnp.sum(_eval(orig, params, param_map, obs_dict, memo))

            case e.Debug(tag, orig):
                result = _eval(orig, params, param_map, obs_dict, memo)
                print(tag, result)

            case _:
                raise ValueError(f"Unknown expression: {expr}")
        memo[id(expr)] = result
        return result
