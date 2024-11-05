import { test } from "vitest";
import { BulgingSegment, LineSegment } from "./segments";

import { Dilatation } from "./sdf-operations";

import {
  expectASCIIDistance,
  expectASCIIshape,
  expectASCIISolidAngle,
} from "./test-utils";
import { asVec } from "./geom";
import { asNum } from "./num";

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
