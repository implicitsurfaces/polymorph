import { test } from "vitest";

import { Dilatation } from "./sdf-operations";

import { expectASCIIshape } from "./test-utils";
import { angleFromDeg, asVec } from "./geom";
import { asNum } from "./num";
import {
  biarcC,
  biarcS,
  bulgingSegmentUsingEndTangent,
  bulgingSegmentUsingStartTangent,
  endpointsEllipticArc,
} from "./segments-helpers";
import { Segment } from "./types";
import { OpenPath } from "./profiles";
import { naiveEval } from "./num-tree";

const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

const t = (s: Segment) => new Dilatation(asNum(0.2), s);

const ot = (s: Segment[]) => new Dilatation(asNum(0.1), new OpenPath(s));

test("bulging from start tangent", async () => {
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingStartTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(30),
        ),
      ),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingStartTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(95),
        ),
      ),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingStartTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(-12),
        ),
      ),
    )
  ).toMatchSnapshot();

  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingStartTangent(
          p(0.5, 0.5),
          p(-0.5, -0.5),
          angleFromDeg(180),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("bulging from end tangent", async () => {
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingEndTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(30),
        ),
      ),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingEndTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(95),
        ),
      ),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingEndTangent(
          p(-0.5, -0.5),
          p(0.5, 0.5),
          angleFromDeg(-12),
        ),
      ),
    )
  ).toMatchSnapshot();

  (
    await expectASCIIshape(
      t(
        bulgingSegmentUsingEndTangent(
          p(0.5, 0.5),
          p(-0.5, -0.5),
          angleFromDeg(180),
        ),
      ),
    )
  ).toMatchSnapshot();
});

test("biarcC", async () => {
  (
    await expectASCIIshape(ot(biarcC(p(-0.5, -0.5), p(0.5, 0.5), p(-0.5, 1))))
  ).toMatchSnapshot();
  (
    await expectASCIIshape(ot(biarcC(p(-0.5, -0.5), p(0.5, 0.5), p(0.5, -1))))
  ).toMatchSnapshot();
  (
    await expectASCIIshape(ot(biarcC(p(-0.5, -0.5), p(0.5, 0.5), p(0.1, -0.1))))
  ).toMatchSnapshot();
  (
    await expectASCIIshape(ot(biarcC(p(-0.5, -0.5), p(0.5, 0.5), p(-1, 1))))
  ).toMatchSnapshot();
});

test("biarcS", async () => {
  (
    await expectASCIIshape(
      ot(biarcS(p(-0.9, 0), p(0.9, 0), p(-0.5, 0.6), p(0.5, -0.5))),
    )
  ).toMatchSnapshot();
  (
    await expectASCIIshape(
      ot(biarcS(p(-0.9, 0), p(0.9, 0), p(-0.8, 0.8), p(0.5, -0.5))),
    )
  ).toMatchSnapshot();
});

test("endpoint ellipse", async () => {
    (
      await expectASCIIshape(
        t(
          endpointsEllipticArc(
            p(0.6, 0),
            p(-0.6, 0),
            asNum(0.8),
            asNum(0.5),
            angleFromDeg(0),
            asNum(0),
            asNum(1),
          ),
        ),
      ),
    )
    .toMatchSnapshot();
});

test("endpoint ellipse with rotation", async () => {
    (
      await expectASCIIshape(
        t(
          endpointsEllipticArc(
            p(0.6, 0),
            p(-0.6, 0),
            asNum(0.9),
            asNum(0.6),
            angleFromDeg(45),
            asNum(0),
            asNum(0),
          ),
        ),
      ),
    )
    .toMatchSnapshot();
});



test("generic endpoint ellipse", async () => {
    (
      await expectASCIIshape(
        t(
          endpointsEllipticArc(
            p(-0.6, -0.2),
            p(0.6, 0.5),
            asNum(0.9),
            asNum(1.6),
            angleFromDeg(15),
            asNum(0),
            asNum(1),
          ),
        ),
      ),
    )
    .toMatchSnapshot();
});
