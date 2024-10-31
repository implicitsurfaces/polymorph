import { test } from "vitest";
import { BulgingSegment, LineSegment } from "./segments";

import { Dilatation } from "./sdf-operations";

import {
  expectASCIIDistance,
  expectASCIIshape,
  expectASCIISolidAngle,
} from "./test-utils";
import { as_vec } from "./geom";
import { as_num } from "./num";

const p = (x: number, y: number) => as_vec(x, y).point_from_origin();

const t = (s: Segment) => new Dilatation(as_num(0.2), s);

test("line segment distance", () => {
  expectASCIIshape(
    t(new LineSegment(p(-0.5, -0.5), p(0.5, 0.5))),
  ).toMatchSnapshot();

  expectASCIIshape(
    t(new LineSegment(p(-0.5, 0.5), p(0.5, -0.5))),
  ).toMatchSnapshot();

  expectASCIIshape(t(new LineSegment(p(-0.5, 0), p(0.5, 0)))).toMatchSnapshot();
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

test("arc distance with positive bulge", () => {
  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), as_num(0.3))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), as_num(0.3))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, 0.5), p(0.5, -0.5), as_num(0.3))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, 0), p(0.5, 0), as_num(0.3))),
  ).toMatchSnapshot();

  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), as_num(1.6))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), as_num(1.6))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, 0.5), p(0.5, -0.5), as_num(1.6))),
  ).toMatchSnapshot();
  expectASCIIshape(
    t(new BulgingSegment(p(-0.5, 0), p(0.5, 0), as_num(1.6))),
  ).toMatchSnapshot();
});

test("arc solid angle with positive bulge", () => {
  expectASCIISolidAngle(
    new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), as_num(0.3)),
  ).toMatchSnapshot();
  expectASCIISolidAngle(
    new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), as_num(0.3)),
  ).toMatchSnapshot();
});
