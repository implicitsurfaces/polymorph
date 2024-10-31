import { test } from "vitest";
import { LineSegment } from "./segments";

import { Dilatation } from "./sdf-operations";

import { expectASCIIshape, expectASCIISolidAngle } from "./test-utils";
import { as_vec } from "./geom";
import { as_num } from "./num";

const p = (x: number, y: number) => as_vec(x, y).point_from_origin();

test("line segment distance", () => {
  expectASCIIshape(
    new Dilatation(as_num(0.2), new LineSegment(p(-0.5, -0.5), p(0.5, 0.5))),
  ).toMatchSnapshot();

  expectASCIIshape(
    new Dilatation(as_num(0.2), new LineSegment(p(-0.5, 0.5), p(0.5, -0.5))),
  ).toMatchSnapshot();

  expectASCIIshape(
    new Dilatation(as_num(0.2), new LineSegment(p(-0.5, 0), p(0.5, 0))),
  ).toMatchSnapshot();
});

test("line segment solid angle", () => {
  expectASCIISolidAngle(
    new LineSegment(p(-0.5, -0.5), p(0.5, 0.5)),
  ).toMatchSnapshot();
  expectASCIISolidAngle(
    new LineSegment(p(0.5, 0.5), p(-0.5, -0.5)),
  ).toMatchSnapshot();

  expectASCIISolidAngle(
    new LineSegment(p(0, 0.5), p(0, -0.5)),
  ).toMatchSnapshot();
});
