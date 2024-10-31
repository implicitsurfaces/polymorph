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
import { angle_from_deg, as_vec } from "./geom";
import { as_num } from "./num";

test("translate", () => {
  const box = new Box(0.5, 0.5);
  expectASCIIshape(new Translation(as_vec(0.5, 0), box)).toMatchSnapshot(
    "right",
  );
  expectASCIIshape(new Translation(as_vec(-0.5, 0), box)).toMatchSnapshot(
    "left",
  );
  expectASCIIshape(new Translation(as_vec(0, 0.5), box)).toMatchSnapshot("top");
  expectASCIIshape(new Translation(as_vec(0, -0.5), box)).toMatchSnapshot(
    "bottom",
  );
});

test("rotate", () => {
  const box = new Translation(as_vec(0.25, 0), new Box(0.5, 0.3));
  expectASCIIshape(new Rotation(angle_from_deg(0), box)).toMatchSnapshot("0");
  expectASCIIshape(new Rotation(angle_from_deg(90), box)).toMatchSnapshot("0");
  expectASCIIshape(new Rotation(angle_from_deg(45), box)).toMatchSnapshot("0");
});

test("scale", () => {
  const box = new Box(0.5, 0.5);
  expectASCIIshape(new Scaling(as_num(2), box)).toMatchSnapshot("bigger");
  expectASCIIshape(new Scaling(as_num(0.5), box)).toMatchSnapshot("smaller");
});

test("dilate", () => {
  const box = new Box(0.5, 0.3);
  expectASCIIshape(new Dilatation(as_num(0.2), box)).toMatchSnapshot("outside");
  expectASCIIshape(new Dilatation(as_num(-0.1), box)).toMatchSnapshot("inside");
});

test("shell", () => {
  const box = new Box(1.3, 0.7);
  expectASCIIshape(new Shell(as_num(0.1), box)).toMatchSnapshot();
});

test("morph", () => {
  const box = new Box(1.8, 0.5);
  const circle = new Circle(0.5);

  expectASCIIshape(new Morph(as_num(0.5), box, circle)).toMatchSnapshot(
    "halfway",
  );
  expectASCIIshape(new Morph(as_num(0.9), box, circle)).toMatchSnapshot(
    "mostly circle",
  );
  expectASCIIshape(new Morph(as_num(0.1), box, circle)).toMatchSnapshot(
    "mostly box",
  );
});

test("union", () => {
  const box = new Box(0.3, 1.9);
  const box2 = new Box(1.9, 0.3);
  const circle = new Circle(0.5);

  expectASCIIshape(new Union(box, circle)).toMatchSnapshot();
  expectASCIIshape(new Union(box, circle, box2)).toMatchSnapshot(
    "union of three",
  );
});

test("intersection", () => {
  const box = new Box(0.7, 1.9);
  const circle = new Circle(0.5);

  expectASCIIshape(new Intersection(box, circle)).toMatchSnapshot();
  expectASCIIshape(
    new Intersection(box, circle, new TopHalfPlane()),
  ).toMatchSnapshot("intersection of three");
});

test("difference", () => {
  const box = new Box(1.9, 0.7);
  const circle = new Circle(0.5);

  expectASCIIshape(new Difference(circle, box)).toMatchSnapshot("remove box");
  expectASCIIshape(new Difference(box, circle)).toMatchSnapshot(
    "remove circle",
  );
});
