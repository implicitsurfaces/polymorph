import jax
import numpy as np
import polymorph_num.expr as e


def np_eval(
    expr: e.Expr, params=None, param_map=None, obs_dict=None, random_key=1, memo=None
) -> np.ndarray:
    if memo is None:
        memo = {}
    if param_map is None:
        param_map = {}
    if obs_dict is None:
        obs_dict = {}

    if expr in memo:
        return memo[expr]

    result = None
    match expr:
        case e.Scalar(value):
            result = np.array(value)

        case e.Arr(value):
            result = np.array(value)

        case e.GridX(width, height):
            half_width = width / 2
            half_height = height / 2

            yy, xx = np.mgrid[-half_height:half_height, -half_width:half_width]
            result = xx.ravel()

        case e.GridY(width, height):
            half_width = width / 2
            half_height = height / 2

            yy, xx = np.mgrid[-half_height:half_height, -half_width:half_width]
            result = yy.ravel()

        case e.GridX3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = xx.ravel()

        case e.GridY3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = yy.ravel()

        case e.GridZ3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = zz.ravel()

        case e.Random(dim, low, high):
            min_ = np_eval(low, params, param_map, obs_dict, random_key, memo)
            max_ = np_eval(high, params, param_map, obs_dict, random_key, memo)

            result = np.random.uniform(size=(dim,), low=min_, high=max_)

        case e.Broadcast(orig, _dim):
            result = np_eval(orig, params, param_map, obs_dict, random_key, memo)

        case e.Unary(orig, op, constants):
            o = np_eval(orig, params, param_map, obs_dict, random_key, memo)
            match op:
                case e.UnOp.Sqrt:
                    result = np.sqrt(o)
                case e.UnOp.Sigmoid:
                    result = jax.nn.sigmoid(o)
                case e.UnOp.SmoothAbs:
                    result = o * np.tanh(10.0 * o)
                case e.UnOp.Abs:
                    result = np.abs(o)
                case e.UnOp.SoftPlus:
                    result = jax.nn.softplus(50 * o) / 50
                case e.UnOp.Log:
                    result = np.log(o)
                case e.UnOp.Exp:
                    result = np.exp(o)
                case e.UnOp.Cos:
                    result = np.cos(o)
                case e.UnOp.Sin:
                    result = np.sin(o)
                case e.UnOp.Sign:
                    result = np.sign(o)
                case e.UnOp.Tanh:
                    result = np.tanh(o)
                case e.UnOp.ArcTan:
                    result = np.arctan(o)
                case e.UnOp.Boxcar:
                    min, max = constants
                    result = np.where((o >= min) & (o <= max), 1.0, 0.0)

        case e.Binary(left, right, op):
            l = np_eval(left, params, param_map, obs_dict, random_key, memo)
            r = np_eval(right, params, param_map, obs_dict, random_key, memo)
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
                    result = np.minimum(l, r)
                case e.BinOp.Max:
                    result = np.maximum(l, r)
                case e.BinOp.ArcTan2:
                    result = np.arctan2(l, r)
                case e.BinOp.Mod:
                    result = np.mod(l, r)

        case e.ComparisonIf(a, b, condition_true, condition_false, op):
            a_ = np_eval(a, params, param_map, obs_dict, random_key, memo=memo)
            b_ = np_eval(b, params, param_map, obs_dict, random_key, memo=memo)
            if op == e.ComparisonOp.Gt:
                result = np.where(
                    a_ > b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )
            elif op == e.ComparisonOp.Ge:
                result = np.where(
                    a_ >= b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )
            elif op == e.ComparisonOp.Eq:
                result = np.where(
                    a_ == b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )

        case e.Param():
            result = param_map.get(expr, params)

        case e.Observation(name):
            result = obs_dict[name]

        case e.Sum(orig):
            result = np.sum(
                np_eval(orig, params, param_map, obs_dict, random_key, memo)
            )

        case e.Debug(tag, orig):
            result = np_eval(orig, params, param_map, obs_dict, random_key, memo)
            print(tag, result)

        case _:
            raise ValueError(f"Unknown expression: {expr}")
    assert result is not None
    memo[expr] = result
    return result
