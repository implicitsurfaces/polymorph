import { test } from "vitest";
import {
  arcWindingNumberIndefiniteIntegral,
  BulgingSegment,
  EllipseArcSegment,
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

test("axis aligned ellipse", async () => {
  (
    await expectASCIIshape(
      t(
        new EllipseArcSegment(
          asNum(0.8),
          asNum(0.4),
          angleFromDeg(10),
          angleFromDeg(120),
          asNum(1),
          p(0, 0),
          angleFromDeg(0),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("axis aligned ellipse, inverse rotation", async () => {
  (
    await expectASCIIshape(
      t(
        new EllipseArcSegment(
          asNum(0.8),
          asNum(0.4),
          angleFromDeg(10),
          angleFromDeg(120),
          asNum(-1),
          p(0, 0),
          angleFromDeg(0),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("rotated ellipse", async () => {
  (
    await expectASCIIshape(
      t(
        new EllipseArcSegment(
          asNum(0.6),
          asNum(0.4),
          angleFromDeg(10),
          angleFromDeg(170),
          asNum(1),
          p(0, 0),
          angleFromDeg(45),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("translated ellipse", async () => {
  (
    await expectASCIIshape(
      t(
        new EllipseArcSegment(
          asNum(0.6),
          asNum(0.2),
          angleFromDeg(10),
          angleFromDeg(170),
          asNum(1),
          p(-0.2, 0.2),
          angleFromDeg(0),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("translated and rotated ellipse", async () => {
  (
    await expectASCIIshape(
      t(
        new EllipseArcSegment(
          asNum(0.6),
          asNum(0.2),
          angleFromDeg(10),
          angleFromDeg(170),
          asNum(1),
          p(-0.2, 0.2),
          angleFromDeg(45),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("ellipse solid angle", async () => {
  (
    await expectASCIISolidAngle(
      new EllipseArcSegment(
        asNum(0.6),
        asNum(0.2),
        angleFromDeg(20),
        angleFromDeg(210),
        asNum(1),
        p(0, 0),
        angleFromDeg(0),
      ),
    )
  ).toMatchSnapshot();
});
