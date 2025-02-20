# Description: This script solves the generic ellipse equation using the Lagrangian method.
#
# You can run it quickly with:
# uv run --no-project --with sympy helpers/solve_generic_ellipse.py


from sympy import (
    Add,
    Float,
    Integer,
    Mul,
    Poly,
    Pow,
    Symbol,
    expand,
    fraction,
    simplify,
    solve,
    symbols,
)
from sympy.abc import A, B, C, D, E, F

# Define our variables
x, y, l = symbols("x y l")
x0, y0 = symbols("x0 y0")

NUM_MAP = {
    16.0: "_16",
    -16.0: "_16neg",
    32.0: "_32",
    -32.0: "_32neg",
    1.0: "ONE",
    -1.0: "NEG_ONE",
    2.0: "TWO",
    -2.0: "_2neg",
    4.0: "_4",
    -4: "_4neg",
    8.0: "_8",
    -8.0: "_8neg",
}


def convert(expr) -> str:
    """Convert a SymPy expression to a string of num operations."""
    if isinstance(expr, (Integer, Float)):
        v = float(expr)
        if v in NUM_MAP:
            return NUM_MAP[v]
        return f"asNum({v})"
    elif isinstance(expr, Symbol):
        return f"{str(expr).lower()}"
    elif isinstance(expr, Add):
        parts = [convert(arg) for arg in expr.args]
        result = parts[0]
        for part in parts[1:]:
            result = f"{result}.add({part})"
        return result
    elif isinstance(expr, Mul):
        parts = [convert(arg) for arg in expr.args]
        result = parts[0]
        for part in parts[1:]:
            result = f"{result}.mul({part})"
        return result
    elif isinstance(expr, Pow):
        base, exp = expr.args
        if exp == 2:
            return f"{convert(base)}.square()"
        if exp == 3:
            return f"{convert(base)}.square().mul({convert(base)})"
        if exp == 4:
            return f"{convert(base)}.square().square()"
        elif exp == -1:
            return f"num(1).div({convert(base)})"
        else:
            raise ValueError(f"Unsupported power operation: {exp}")
    else:
        raise ValueError(f"Unsupported operation: {type(expr)}")


def print_coeffs_as_num(expr, variable):
    poly = Poly(expr.simplify(), l)
    coeffs = poly.coeffs()

    print(f"const l4 = {convert(coeffs[4])};\n")
    print(f"const l3 = {convert(coeffs[3])};\n")
    print(f"const l2 = {convert(coeffs[2])};\n")
    print(f"const l1 = {convert(coeffs[1])};\n")
    print(f"const l0 = {convert(coeffs[0])};\n")


def solve_with_lagrangian():
    # Define the Lagrangian

    constraint = A * x**2 + 2 * B * x * y + C * y**2 + 2 * D * x + 2 * E * y + F
    L = (x - x0) ** 2 + (y - y0) ** 2 + l * constraint

    # Take partial derivatives
    dL_dx = L.diff(x)
    dL_dy = L.diff(y)

    # Solve for x and y
    system = [dL_dx, dL_dy]
    solution = solve(system, [x, y])

    # Extract x(λ) and y(λ)
    x_lambda = simplify(solution[x])
    y_lambda = simplify(solution[y])

    # Substitute these expressions into the constraint equation

    # Get the final equation in λ by substituting x(λ) and y(λ)
    final_eq = constraint.subs([(x, x_lambda), (y, y_lambda)])
    final_eq = simplify(expand(final_eq))

    num, denom = fraction(final_eq)

    print_coeffs_as_num(num, l)

    print(f"const x = {convert(simplify(x_lambda))};\n")
    print(f"const y = {convert(simplify(y_lambda))};\n")


if __name__ == "__main__":
    solve_with_lagrangian()
