import math

import polymorph_s2df as s2df
from hypothesis import assume, given
from hypothesis import strategies as st
from polymorph_num.ops import observation
from polymorph_num.unit import Unit
from polymorph_s2df.operations import Shape
from polymorph_s2df.solid_operations import Solid

wedge_with_arc = s2df.draw((-1, -0.5)).arc(1, 0, 0.4).arc(0, 1, -1.2).close()
sweeped_solid = (
    s2df.sweep(wedge_with_arc)
    .to_point((0.5, 0.3, 0.8))
    .extrude(0.5)
    .morph(s2df.Box(0.5, 1))
    .to_solid()
)


def distance_gradient(profile: Shape):
    distance = profile.distance(observation("x"), observation("y"))

    unit = Unit(["x", "y"]).register("grad_distance", distance, grad=True).compile()

    def grad_fn(x, y):
        _, obs = unit.observe(dict(x=x, y=y)).evaluate("grad_distance")
        return obs

    return grad_fn


def distance_gradient_3d(profile: Solid):
    distance = profile.distance(observation("x"), observation("y"), observation("z"))

    unit = (
        Unit(["x", "y", "z"]).register("grad_distance", distance, grad=True).compile()
    )

    def grad_fn(x, y, z):
        _, obs = unit.observe(dict(x=x, y=y, z=z)).evaluate("grad_distance")
        return obs

    return grad_fn


def inside_gradient(profile: Shape):
    distance = profile.is_inside(observation("x"), observation("y"))

    unit = Unit(["x", "y"]).register("grad_inside", distance, grad=True).compile()

    def grad_fn(x, y):
        _, obs = unit.observe(dict(x=x, y=y)).evaluate("grad_inside")
        return obs

    return grad_fn


def inside_gradient_3d(profile: Solid):
    distance = profile.is_inside(observation("x"), observation("y"), observation("z"))

    unit = Unit(["x", "y", "z"]).register("grad_inside", distance, grad=True).compile()

    def grad_fn(x, y, z):
        _, obs = unit.observe(dict(x=x, y=y, z=z)).evaluate("grad_inside")
        return obs

    return grad_fn


distance_gradient_sweeped_solid = distance_gradient_3d(sweeped_solid)
inside_gradient_sweeped_solid = inside_gradient_3d(sweeped_solid)


distance_gradient_arc = distance_gradient(s2df.ArcSegment(0.5, 2.2, 0.5, 1))


@given(x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_arc_gradients(x, y):
    # gradients is not defined at the center of the arc
    assume(abs(x) > 1e-12 and abs(y) > 1e-12)

    grad = distance_gradient_arc(x, y)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])


distance_gradient_path = distance_gradient(wedge_with_arc)


@given(x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_path_gradients(x, y):
    # gradients are not defined at the corners of the wedge
    assume(abs(x + 1) > 1e-12 and abs(y + 0.5) > 1e-12)
    assume(abs(x - 1) > 1e-12 and abs(y) > 1e-12)
    assume(abs(x) > 1e-12 and abs(y - 1) > 1e-12)

    grad = distance_gradient_path(x, y)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])


inside_gradient_path = inside_gradient(wedge_with_arc)


@given(x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_path_inside_gradient(x, y):
    grad = inside_gradient_path(x, y)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])


extruded_solid = s2df.embed_in_3d(
    wedge_with_arc, s2df.XY_PLANE.translate(x=0.1, z=-0.2)
).extrude(1)
distance_gradient_extruded_solid = distance_gradient_3d(extruded_solid)


@given(x=st.floats(-2, 2), y=st.floats(-2, 2), z=st.floats(-2, 2))
def test_extruded_solid_gradient(x, y, z):
    assume(abs(z) > 1e-12)
    grad = distance_gradient_extruded_solid(x, y, z)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])
    assert not math.isnan(grad["z"])


inside_gradient_extruded_solid = distance_gradient_3d(extruded_solid)


@given(x=st.floats(-2, 2), y=st.floats(-2, 2), z=st.floats(-2, 2))
def test_extruded_solid_inside_gradient(x, y, z):
    grad = inside_gradient_extruded_solid(x, y, z)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])
    assert not math.isnan(grad["z"])


@given(x=st.floats(-2, 2), y=st.floats(-2, 2), z=st.floats(-2, 2))
def test_sweeped_solid_gradients(x, y, z):
    # gradients are not defined on the surface of the solid
    assume(abs(z) > 1e-12)

    grad = distance_gradient_sweeped_solid(x, y, z)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])
    assert not math.isnan(grad["z"])


@given(x=st.floats(-2, 2), y=st.floats(-2, 2), z=st.floats(-2, 2))
def test_sweeped_solid_inside_gradient(x, y, z):
    grad = inside_gradient_sweeped_solid(x, y, z)
    assert not math.isnan(grad["x"])
    assert not math.isnan(grad["y"])
    assert not math.isnan(grad["z"])


def test_distance_to_arc_segment(benchmark):
    unit = (
        Unit(["x", "y"])
        .register(
            "distance",
            s2df.ArcSegment(0.5, 2.2, 0.5, 1).distance(
                observation("x"), observation("y")
            ),
        )
        .compile()
    )

    def arc_segment_distance():
        return unit.observe({"x": 1.2, "y": 1.3}).evaluate("distance")

    benchmark(arc_segment_distance)


def test_distance_to_bulding_arc_segment(benchmark):
    unit = (
        Unit(["x", "y"])
        .register(
            "distance",
            s2df.bulge_arc((0.5, 0.5), (-1, 0.2), 0.8).distance(
                observation("x"), observation("y")
            ),
        )
        .compile()
    )

    def arc_segment_distance():
        return unit.observe({"x": 1.2, "y": 1.3}).evaluate("distance")

    benchmark(arc_segment_distance)


def test_distance_to_bulging_segment(benchmark):
    unit = (
        Unit(["x", "y"])
        .register(
            "distance",
            s2df.BulgingSegment((0.5, 0.5), (-1, 0.2), 0.8).distance(
                observation("x"), observation("y")
            ),
        )
        .compile()
    )

    def arc_segment_distance():
        return unit.observe({"x": 1.2, "y": 1.3}).evaluate("distance")

    benchmark(arc_segment_distance)
