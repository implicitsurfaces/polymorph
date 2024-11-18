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

test("circle", async () => {
  (await expectASCIIshape(new Circle(0.8))).toMatchSnapshot();
  (await expectASCIIshape(new Circle(0.1))).toMatchSnapshot();
  (await expectASCIIDistance(new Circle(0.8))).toMatchSnapshot();
});

test("box", async () => {
  (await expectASCIIshape(new Box(1.7, 1))).toMatchSnapshot();
  (await expectASCIIshape(new Box(0.2, 1.8))).toMatchSnapshot();
});

test("top half plane", async () => {
  (await expectASCIIshape(new TopHalfPlane())).toMatchSnapshot();
});

test("bottom half plane", async () => {
  (await expectASCIIshape(new BottomHalfPlane())).toMatchSnapshot();
});

test("left half plane", async () => {
  (await expectASCIIshape(new LeftHalfPlane())).toMatchSnapshot();
});

test("right half plane", async () => {
  (await expectASCIIshape(new RightHalfPlane())).toMatchSnapshot();
});

test("closed path", async () => {
  (
    await expectASCIIDistance(
      new ClosedPath([
        new LineSegment(p(-0.5, -0.5), p(0.5, 0)),
        new BulgingSegment(p(0.5, 0), p(0.7, 0.7), asNum(0.5)),
        new LineSegment(p(0.7, 0.7), p(0, 0.8)),
        new LineSegment(p(0, 0.8), p(-0.5, -0.5)),
      ]),
    )
  ).toMatchSnapshot();
});

test("basic heart shape", async () => {
  const p0 = asVec(0, -0.8).pointFromOrigin();
  const p1 = asVec(-0.9, 0.3).pointFromOrigin();
  const p2 = asVec(0, 0.3).pointFromOrigin();
  const p3 = asVec(0.9, 0.3).pointFromOrigin();

  const heart = new ClosedPath([
    new LineSegment(p0, p1),
    new BulgingSegment(p1, p2, asNum(-0.9)),
    new BulgingSegment(p2, p3, asNum(-0.9)),
    new LineSegment(p3, p0),
  ]);

  (await expectASCIIDistance(heart)).toMatchSnapshot();
});
