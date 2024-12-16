import { test } from "vitest";
import {
  arcWindingNumberIndefiniteIntegral,
  BulgingSegment,
  LineSegment,
} from "./segments";

import { Dilatation } from "./sdf-operations";

import {
  ex,
  expectASCIIDistance,
  expectASCIIshape,
  expectASCIISolidAngle,
} from "./test-utils";
import { angleFromDeg, asVec } from "./geom";
import { asNum } from "./num";
import { Segment } from "./types";

const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

const t = (s: Segment) => new Dilatation(asNum(0.2), s);

test("line segment distance", async () => {
  (
    await expectASCIIshape(t(new LineSegment(p(-0.5, -0.5), p(0.5, 0.5))))
  ).toMatchSnapshot();

  (
    await expectASCIIshape(t(new LineSegment(p(-0.5, 0.5), p(0.5, -0.5))))
  ).toMatchSnapshot();

  (
    await expectASCIIshape(t(new LineSegment(p(-0.5, 0), p(0.5, 0))))
  ).toMatchSnapshot();
});

test("line segment solid angle", async () => {
  (
    await expectASCIISolidAngle(new LineSegment(p(-0.5, -0.5), p(0.5, 0.5)))
  ).toMatchSnapshot();
  (
    await expectASCIISolidAngle(new LineSegment(p(0.5, 0.5), p(-0.5, -0.5)))
  ).toMatchSnapshot();

  (
    await expectASCIISolidAngle(new LineSegment(p(0, 0.5), p(0, -0.5)))
  ).toMatchSnapshot();
});

test("arc distance with positive bulge", async () => {
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), asNum(0.3))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), asNum(0.3))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, 0.5), p(0.5, -0.5), asNum(0.3))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, 0), p(0.5, 0), asNum(0.3))),
    )
  ).toMatchSnapshot();

  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), asNum(1.6))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), asNum(1.6))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, 0.5), p(0.5, -0.5), asNum(1.6))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(new BulgingSegment(p(-0.5, 0), p(0.5, 0), asNum(1.6))),
    )
  ).toMatchSnapshot();
});

test("arc solid angle with positive bulge", async () => {
  (
    await expectASCIISolidAngle(
      new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), asNum(0.3)),
    )
  ).toMatchSnapshot();
  (
    await expectASCIISolidAngle(
      new BulgingSegment(p(0.5, 0.5), p(-0.5, -0.5), asNum(0.3)),
    )
  ).toMatchSnapshot();
});

test("arc solid angle with bulge > 1", async () => {
  (
    await expectASCIISolidAngle(
      new BulgingSegment(p(-0.5, -0.5), p(0.5, 0.5), asNum(1.3)),
    )
  ).toMatchSnapshot();
});

test("winding number computation", async () => {
  ex(
    arcWindingNumberIndefiniteIntegral(
      angleFromDeg(135),
      asNum(1.2),
      asNum(0),
      asNum(-0.2),
    ).turns,
  ).toBeCloseTo(0.6346);

  ex(
    arcWindingNumberIndefiniteIntegral(
      angleFromDeg(45),
      asNum(1.2),
      asNum(0),
      asNum(-0.2),
    ).turns,
  ).toBeCloseTo(0.418);
});

test("arc pill", async () => {
  const p0 = asVec(-0.5, -0.3).pointFromOrigin();
  const p1 = asVec(0.3, 0.5).pointFromOrigin();
  const pill = t(new BulgingSegment(p0, p1, asNum(0.9)));

  (await expectASCIIDistance(pill)).toMatchSnapshot();
});
