import sympy as sp

# Define symbolic variables
a, b, x, y, t, lam = sp.symbols("a b x y t lambda", real=True)

print("SYMBOLIC VERIFICATION USING SYMPY\n")
print("1. Define the parametric equation for a hyperbola:")
q_x = a * sp.cosh(t)
q_y = b * sp.sinh(t)
print(f"   q(t) = ({q_x}, {q_y})\n")

# Calculate squared distance function
print("2. Define squared distance function from point (x,y) to hyperbola:")
dist_squared = (q_x - x) ** 2 + (q_y - y) ** 2
dist_squared_expanded = sp.expand(dist_squared)
print(f"   s²(t) = {dist_squared}")
print(f"   s²(t) = {dist_squared_expanded}\n")

# Calculate the derivative with respect to t
print("3. Calculate derivative with respect to t and set to zero:")
derivative = sp.diff(dist_squared, t)
derivative_simplified = sp.simplify(derivative)

# Factor and further simplify the derivative
print("4. Simplify the derivative equation:")
factored_derivative = sp.factor(derivative_simplified)
print(f"   {factored_derivative} = 0\n")

# Make the substitution lambda = cosh(t)
print("5. Substitute λ = cosh(t), noting that sinh(t) = √(λ² - 1):")
# We'll manually work with the simplified form
substituted = factored_derivative.subs(sp.cosh(t), lam).subs(
    sp.sinh(t), sp.sqrt(lam**2 - 1)
)
print(f"   {substituted} = 0\n")

print("6. For λ > 1, divide by √(λ² - 1):")
divided = (substituted) / sp.sqrt(lam**2 - 1)
print(f"   {divided} = 0\n")


print(sp.simplify(divided))


# Square both sides
print("7. Square both sides to eliminate the radical:")
print("   ((a² + b²)·λ - a·x)² = (b·y·λ)²/(λ² - 1)")
print("   Multiply both sides by (λ² - 1):")
print("   ((a² + b²)·λ - a·x)²·(λ² - 1) = (b·y·λ)²\n")

# Set up the algebraic equation after squaring
left_side = ((a**2 + b**2) * lam - a * x) ** 2 * (lam**2 - 1)
right_side = (b * y * lam) ** 2

print("8. Expand both sides of the equation:")
left_expanded = sp.expand(left_side)
right_expanded = sp.expand(right_side)
print(f"   Left side: {left_expanded}")
print(f"   Right side: {right_expanded}\n")


# Set up the quartic equation
print("9. Set left side - right side = 0 to get the quartic equation:")
quartic = left_expanded - right_expanded
quartic_equation = sp.collect(sp.expand(quartic), lam)
print(f"   {quartic_equation} = 0\n")

# Collect terms by power of lambda
print("10. Collect terms by power of λ:")
collected = sp.collect(quartic_equation, lam, evaluate=False)
poly = sp.Poly(quartic_equation, lam)
coeffs = poly.all_coeffs()
coeffs.reverse()
for degree, ex in enumerate(coeffs):
    if degree != 0:
        print(f"   λ^{degree} term: {ex}")
    else:
        print(f"   Constant term: {ex}")
print()

# Define m and n² for normalization
print("11. Define substitution variables:")
m_expr = a * x / (a**2 + b**2)

# Calculate the n² expression
a_term = (a * x) ** 2
b_term = (b * y) ** 2
c_term = -((a**2 + b**2) ** 2)
n_squared_num = a_term + b_term + c_term
n_squared_den = (a**2 + b**2) ** 2
n_squared = n_squared_num / n_squared_den
