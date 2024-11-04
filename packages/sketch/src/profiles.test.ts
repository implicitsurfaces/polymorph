import { test } from "vitest";
import {
  Circle,
  Box,
  TopHalfPlane,
  BottomHalfPlane,
  LeftHalfPlane,
  RightHalfPlane,
  ClosedPath,
} from "./profiles";

import { expectASCIIshape, expectASCIIDistance } from "./test-utils";
import { BulgingSegment, LineSegment } from "./segments";
import { asVec } from "./geom";
import { asNum } from "./num";

const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

test("circle", () => {
  expectASCIIshape(new Circle(0.8)).toMatchSnapshot();
  expectASCIIshape(new Circle(0.1)).toMatchSnapshot();
  expectASCIIDistance(new Circle(0.8)).toMatchSnapshot();
});

test("box", () => {
  expectASCIIshape(new Box(1.7, 1)).toMatchSnapshot();
  expectASCIIshape(new Box(0.2, 1.8)).toMatchSnapshot();
});

test("top half plane", () => {
  expectASCIIshape(new TopHalfPlane()).toMatchSnapshot();
});

test("bottom half plane", () => {
  expectASCIIshape(new BottomHalfPlane()).toMatchSnapshot();
});

test("left half plane", () => {
  expectASCIIshape(new LeftHalfPlane()).toMatchSnapshot();
});

test("right half plane", () => {
  expectASCIIshape(new RightHalfPlane()).toMatchSnapshot();
});

test("closed path", () => {
  expectASCIIDistance(
    new ClosedPath([
      new LineSegment(p(-0.5, -0.5), p(0.5, 0)),
      new BulgingSegment(p(0.5, 0), p(0.7, 0.7), asNum(0.5)),
      new LineSegment(p(0.7, 0.7), p(0, 0.8)),
      new LineSegment(p(0, 0.8), p(-0.5, -0.5)),
    ]),
  ).toMatchSnapshot();
});
