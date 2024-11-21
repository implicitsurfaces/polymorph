import { test } from "vitest";

import { Dilatation } from "./sdf-operations";

import { expectASCIIDistance } from "./test-utils";
import { asVec } from "./geom";
import { asNum } from "./num";
import {
  filletArcArc,
  filletArcLine,
  filletLineArc,
  filletLineLine,
} from "./segments-fillets";
import { Segment } from "./types";
import { OpenPath } from "./profiles";
import { BulgingSegment, LineSegment } from "./segments";

const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

const d = (s: Segment[]) => new Dilatation(asNum(0.1), new OpenPath(s));

test("fillet line line", async () => {
  const p0 = p(-0.5, -0.5);
  const p1 = p(0.1, 0.2);
  const p2 = p(0.5, -0.5);

  const l1 = new LineSegment(p0, p1);
  const l2 = new LineSegment(p1, p2);

  (
    await expectASCIIDistance(d(filletLineLine(l1, l2, asNum(0.2))))
  ).toMatchSnapshot();

  const p3 = p(-0.2, 0.5);

  const l3 = new LineSegment(p1, p3);

  (
    await expectASCIIDistance(d(filletLineLine(l1, l3, asNum(0.2))))
  ).toMatchSnapshot();

  const l4 = new LineSegment(p2, p1);
  const l5 = new LineSegment(p1, p0);

  (
    await expectASCIIDistance(d(filletLineLine(l4, l5, asNum(0.2))))
  ).toMatchSnapshot();
});

test("fillet line arc", async () => {
  const p0 = p(-0.5, -0.5);
  const p1 = p(0.1, 0.2);
  const p2 = p(0.5, -0.5);

  const l1 = new LineSegment(p0, p1);
  const l2 = new BulgingSegment(p1, p2, asNum(0.3));

  (
    await expectASCIIDistance(d(filletLineArc(l1, l2, asNum(0.2))))
  ).toMatchSnapshot();

  const p3 = p(-1.2, 0.5);

  const l3 = new BulgingSegment(p1, p3, asNum(0.3));

  (
    await expectASCIIDistance(d(filletLineArc(l1, l3, asNum(0.2))))
  ).toMatchSnapshot();

  const l4 = new LineSegment(p2, p1);
  const l5 = new BulgingSegment(p1, p0, asNum(0.3));

  (
    await expectASCIIDistance(d(filletLineArc(l4, l5, asNum(0.2))))
  ).toMatchSnapshot();
});

test("fillet arc line", async () => {
  const p0 = p(-0.5, -0.5);
  const p1 = p(0.1, 0.2);
  const p2 = p(0.5, -0.5);

  const l1 = new BulgingSegment(p0, p1, asNum(0.3));
  const l2 = new LineSegment(p1, p2);

  (
    await expectASCIIDistance(d(filletArcLine(l1, l2, asNum(0.2))))
  ).toMatchSnapshot();
});

test("fillet arc arc", async () => {
  const p0 = p(-0.5, -0.5);
  const p1 = p(0.1, 0.2);
  const p2 = p(0.5, -0.5);

  const l1 = new BulgingSegment(p0, p1, asNum(-0.3));
  const l2 = new BulgingSegment(p1, p2, asNum(0.4));

  (await expectASCIIDistance(d([l1, l2]))).toMatchSnapshot();

  (
    await expectASCIIDistance(d(filletArcArc(l1, l2, asNum(0.2))))
  ).toMatchSnapshot();

  const p3 = p(-1.2, 0.5);

  const l3 = new BulgingSegment(p1, p3, asNum(0.2));

  (
    await expectASCIIDistance(d(filletArcArc(l1, l3, asNum(0.2))))
  ).toMatchSnapshot();

  const l4 = new BulgingSegment(p2, p1, asNum(-0.4));
  const l5 = new BulgingSegment(p1, p0, asNum(0.3));

  (
    await expectASCIIDistance(d(filletArcArc(l4, l5, asNum(0.2))))
  ).toMatchSnapshot();
});
