# Working with Nums

## What is a Num?

Most people should interact with `Num` objects as if they were numbers.

The difference is that instead of being a normal number, it is the basic
building block to create algebraic operations graphs.

We want to create graphs of operations that represent complex computation in
a way that is easy to manipulate and understand. Ideally we would use operator
overloading, but javascript does not offer it. So we need to create an object
that behaves like a number but can be manipulated in a way that we can create
a tree of operations.

The `Num` class is this object that behaves like a number, but builds a graph
of operations instead of performing the operation.

## Methods exposed by a `Num`

Here are the methods exposed on a `Num` object.

### Arithmetic Operations

- `add(other: Num | number)` - Adds a value to this number
- `sub(other: Num | number)` - Subtracts a value from this number
- `mul(other: Num | number)` - Multiplies this number by a value
- `div(other: Num | number)` - Divides this number by a value
- `mod(other: Num)` - Returns the remainder after division by another Num

### Basic Math Operations

- `sqrt()` - Square root operation
- `safeSqrt()` - Square root that first ensures the value is non-negative
- `cbrt()` - Cube root operation
- `neg()` - Negates the number (returns -n)
- `inv()` - Returns the inverse (1/n)
- `sign()` - Returns the sign of the number (-1, 0, or 1)
- `abs()` - Returns the absolute value
- `smoothabs()` - Returns a differentiable approximation of absolute value using tanh
- `square()` - Returns the number squared (n²)

### Exponential & Logarithmic Operations

- `exp()` - Returns e raised to the power of this number
- `log()` - Returns the natural logarithm
- `log1p()` - Returns ln(1+n), numerically stable for small values

### Trigonometric Functions

Note that if you are working with angle you would be better using the `Angle`
object.

- `sin()` - Sine function
- `cos()` - Cosine function
- `tan()` - Tangent function
- `asin()` - Inverse sine (arcsine)
- `acos()` - Inverse cosine (arccosine)
- `atan()` - Inverse tangent (arctangent)
- `tanh()` - Hyperbolic tangent

### Activation Functions

- `softplus()` - Returns a smooth approximation of ReLU: ln(1+e^x)
- `softminus()` - Returns x - softplus(x)

### Comparison Operations

- `compare(other: Num | number)` - Returns a comparison value (-1, 0, 1)
- `equals(other: Num | number)` - Tests equality
- `lessThan(other: Num | number)` - Tests if this number is less than another
- `lessThanOrEqual(other: Num | number)` - Tests if this number is less than or equal to another
- `greaterThan(other: Num | number)` - Tests if this number is greater than another
- `greaterThanOrEqual(other: Num | number)` - Tests if this number is greater than or equal to another
- `max(other: Num | number)` - Returns the maximum of this number and another
- `min(other: Num | number)` - Returns the minimum of this number and another

### Logical Operations

- `and(other: Num | number)` - Lazy logical AND operation (i.e. return other if this is non-zero)
- `or(other: Num | number)` - Lazy logical OR operation (i.e. return this if this is non-zero)
- `not()` - Logical NOT operation

### Utility Methods

- `debug(info: string)` - Adds debug information (useful when looking at the dot graph)
- `compress()` - Returns a compressed version of this Num (i.e. deduplicates nodes)
- `simplify()` - Returns a simplified version of this Num (i.e. evaluates literal expressions)
- `asDot()` - Returns a DOT language representation of the Num node tree

### Additional helper functions

There are some helper functions that are not part of the `Num` class but are
very useful when working with `Num` objects. You can import them from the
`num-oper

- `ifTruthyElse(condition: Num | number, ifNonZero: Num | number, ifZero:
Num | number): Num` - Implements a conditional operation that returns
  `ifNonZero` when `condition` is truthy (non-zero), and `ifZero` when
  `condition` is falsy (zero)

- `hypot(a: Num | number, b: Num | number): Num` - Calculates the hypotenuse
  length of a right triangle given the two other sides (√(a² + b²))

- `clamp(a: Num | number, minVal: Num | number, maxVal: Num | number): Num`
  - Constrains a value to be within a specified range, returning the value if
    it's between min and max, or the min/max bound if the value exceeds the
    range

## Literal and variable `Nums`

In addition to the methods above, you will need to create a `Num` object at
some point. There are two main leaf `Num`:

- a literal number that can be created with `asNum(number)`
- a varible that can be created with `variable(string)`

## Working with geometric higher-level objects

When working with geometric objects like points, vectors, and angles, you
should use the `Point`, `Vector`, and `Angle` classes. These classes are
designed to work with `Num` objects and provide a more intuitive interface for
working with geometric objects.

### Point

The `Point` class represents a point in 2D space. It has the following methods:

#### Construction

- `constructor(private _x: Num, private _y: Num)` - Creates a new Point with the given x and y coordinates

#### Transformation Methods

- `add(vec: Vec2): Point` - Returns a new Point by adding a vector to this point
- `sub(vec: Vec2): Point` - Returns a new Point by subtracting a vector from this point

#### Point Relationships

- `midPoint(other: Point): Point` - Calculates the midpoint between this point and another point
- `vecTo(other: Point): Vec2` - Returns a vector from this point to another point
- `vecFrom(other: Point): Vec2` - Returns a vector from another point to this point
- `vecFromOrigin(): Vec2` - Returns a vector from the origin (0,0) to this point

#### Accessors

- `get x(): Num` - Returns the x-coordinate of this point
- `get y(): Num` - Returns the y-coordinate of this point

### Vec2

#### Construction

- `constructor(x: Num, y: Num)` - Creates a new 2D vector with the given x and y components

#### Vector Operations

- `add(other: Vec2): Vec2` - Returns a new vector by adding another vector to this one
- `sub(other: Vec2): Vec2` - Returns a new vector by subtracting another vector from this one
- `neg(): Vec2` - Returns the negation of this vector (-x, -y)
- `scale(other: Num | number): Vec2` - Multiplies this vector by a scalar value
- `div(other: Num | number): Vec2` - Divides this vector by a scalar value

#### Vector Products

- `dot(other: Vec2): Num` - Calculates the dot product with another vector
- `cross(other: Vec2): Num` - Calculates the 2D cross product with another vector

#### Vector Attributes

- `norm(): Num` - Calculates the length/magnitude of this vector
- `normalize(): UnitVec2` - Returns a unit vector in the same direction as this vector

#### Accessors

- `get x(): Num` - Returns the x-component of this vector
- `get y(): Num` - Returns the y-component of this vector

#### Transformations

- `perp(): Vec2` - Returns the perpendicular vector (rotated 90° counter-clockwise)
- `mirrorX(): Vec2` - Mirrors this vector across the y-axis
- `mirrorY(): Vec2` - Mirrors this vector across the x-axis
- `rotate(angle: Angle): Vec2` - Rotates this vector by the specified angle

#### Conversions

- `asAngle(): Angle` - Converts this vector to an angle (assumes unit vector or normalizes first)
- `pointFromOrigin(): Point` - Creates a point at the position specified by this vector from the origin

Note that there is a `UnitVec2` class that represents a unit vector (a vector
with a magnitude of 1). This allows for more efficient calculations when we
know the vector is a unit vector.

### Angle

The `Angle` class represents an angle in 2D space. It is represented by its
cosine and sine components, which allows to avoid as much as possible the use
of trigonometric functions.

#### Construction

- `constructor(cos: Num, sin: Num)` - Creates a new Angle using cosine and sine components

#### Angle Operations

- `add(other: Angle): Angle` - Adds another angle to this one (equivalent to rotation composition)
- `sub(other: Angle): Angle` - Subtracts another angle from this one
- `neg(): Angle` - Returns the negation of this angle (rotation in opposite direction)
- `half(): Angle` - Returns half of this angle
- `double(): Angle` - Returns double of this angle
- `perp(): Angle` - Returns a perpendicular angle (rotated by 90°)
- `opposite(): Angle` - Returns the opposite angle (rotated by 180°)

#### Trigonometric Components

- `cos(): Num` - Returns the cosine component of this angle
- `sin(): Num` - Returns the sine component of this angle
- `tan(): Num` - Returns the tangent value (sin/cos) of this angle

#### Conversions

- `asRad(): Num` - Converts this angle to radians
- `asDeg(): Num` - Converts this angle to degrees
- `asSortValue(): Num` - Returns a value that can be used for angle sorting/comparison
- `asUnitArcLength(): Num` - Returns the normalized arc length (0 to 2π)
- `asVec(): UnitVec2` - Converts this angle to a unit vector

#### Additional conversion functions

##### Creation from Different Units

- `angleRromRad(rad: Num | number): Angle` - Creates an angle from a radian value
- `angleFromDeg(deg: Num | number): Angle` - Creates an angle from a degree value

##### Creation from Trigonometric Values

- `angleFromSin(sin: Num | number): Angle` - Creates an angle from its sine value
- `angleFromCos(cos: Num | number): Angle` - Creates an angle from its cosine value

##### Creation from Vectors

- `angleFromDirection(direction: Vec2): Angle` - Creates an angle from a direction vector
- `twoVectorsAngle(v1: Vec2, v2: Vec2): Angle` - Creates an angle between two vectors
- `arcTan(x: Num | number, y: Num | number): Angle` - Creates an angle from x and y coordinates (equivalent to atan2, with swapped arguments)

## Evaluation of `Nums`

The `Num` class is designed to build a graph of operations that can be
evaluated. The evaluation is done by calling different function depending on
how the graph is to be evaluated.

### Num and NumNodes

In order to have a clear separation between the data and the tool to create the
data we have two classes: `Num` and `NumNode`.

`Num` is the class that the user will interact with. It is the class that
exposes the methods to create the operations and the methods to manipulate the
data.

`NumNode` is the class that a tree of operations is made of.

You can access the `NumNode` of a `Num` by calling its `.n` property.

### Evaluation with javascript

The evaluation of the graph is done by calling the `simpleEval` method (from
`eval-num/js-eval.ts`).

Note that if you have some variables in your graph, you will need to provide
a `Map` of variable names to values to the `simpleEval` function as a second
argument.

### Evaluation with fidget

You can do a simple eval with fidget by calling the `fidgetEval` function from
`eval-num/fidget-eval.ts`.

You might want to do a fidget render of a class that exposes a `distanceTo`
method by using `fidgetRender` from `eval-num/fidget-render.ts`.
