import { test } from "vitest";
import { Box, Circle, TopHalfPlane } from "./profiles";

import { expectASCIIshape } from "./test-utils";
import {
  Difference,
  Dilatation,
  Intersection,
  Morph,
  Rotation,
  Scaling,
  Shell,
  Translation,
  Union,
} from "./sdf-operations";
import { angleFromDeg, asVec } from "./geom";
import { asNum } from "./num";

test("translate", async () => {
  const box = new Box(0.5, 0.5);
  (await expectASCIIshape(new Translation(asVec(0.5, 0), box))).toMatchSnapshot(
    "right",
  );
  (
    await expectASCIIshape(new Translation(asVec(-0.5, 0), box))
  ).toMatchSnapshot("left");
  (await expectASCIIshape(new Translation(asVec(0, 0.5), box))).toMatchSnapshot(
    "top",
  );
  (
    await expectASCIIshape(new Translation(asVec(0, -0.5), box))
  ).toMatchSnapshot("bottom");
});

test("rotate", async () => {
  const box = new Translation(asVec(0.25, 0), new Box(0.5, 0.3));
  (await expectASCIIshape(new Rotation(angleFromDeg(0), box))).toMatchSnapshot(
    "0",
  );
  (await expectASCIIshape(new Rotation(angleFromDeg(90), box))).toMatchSnapshot(
    "0",
  );
  (await expectASCIIshape(new Rotation(angleFromDeg(45), box))).toMatchSnapshot(
    "0",
  );
});

test("scale", async () => {
  const box = new Box(0.5, 0.5);
  (await expectASCIIshape(new Scaling(asNum(2), box))).toMatchSnapshot(
    "bigger",
  );
  (await expectASCIIshape(new Scaling(asNum(0.5), box))).toMatchSnapshot(
    "smaller",
  );
});

test("dilate", async () => {
  const box = new Box(0.5, 0.3);
  (await expectASCIIshape(new Dilatation(asNum(0.2), box))).toMatchSnapshot(
    "outside",
  );
  (await expectASCIIshape(new Dilatation(asNum(-0.1), box))).toMatchSnapshot(
    "inside",
  );
});

test("shell", async () => {
  const box = new Box(1.3, 0.7);
  (await expectASCIIshape(new Shell(asNum(0.1), box))).toMatchSnapshot();
});

test("morph", async () => {
  const box = new Box(1.8, 0.5);
  const circle = new Circle(0.5);

  (await expectASCIIshape(new Morph(asNum(0.5), box, circle))).toMatchSnapshot(
    "halfway",
  );
  (await expectASCIIshape(new Morph(asNum(0.9), box, circle))).toMatchSnapshot(
    "mostly circle",
  );
  (await expectASCIIshape(new Morph(asNum(0.1), box, circle))).toMatchSnapshot(
    "mostly box",
  );
});

test("union", async () => {
  const box = new Box(0.3, 1.9);
  const box2 = new Box(1.9, 0.3);
  const circle = new Circle(0.5);

  (await expectASCIIshape(new Union(box, circle))).toMatchSnapshot();
  (await expectASCIIshape(new Union(box, circle, box2))).toMatchSnapshot(
    "union of three",
  );
});

test("intersection", async () => {
  const box = new Box(0.7, 1.9);
  const circle = new Circle(0.5);

  (await expectASCIIshape(new Intersection(box, circle))).toMatchSnapshot();
  (
    await expectASCIIshape(new Intersection(box, circle, new TopHalfPlane()))
  ).toMatchSnapshot("intersection of three");
});

test("difference", async () => {
  const box = new Box(1.9, 0.7);
  const circle = new Circle(0.5);

  (await expectASCIIshape(new Difference(circle, box))).toMatchSnapshot(
    "remove box",
  );
  (await expectASCIIshape(new Difference(box, circle))).toMatchSnapshot(
    "remove circle",
  );
});
