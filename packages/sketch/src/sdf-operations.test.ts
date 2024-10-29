import { test } from "vitest";
import { Box, Circle, TopHalfPlane } from "./profiles";

import { expect_ascii_dist } from "./test-utils";
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
  expect_ascii_dist(new Translation(as_vec(0.5, 0), box)).toMatchSnapshot(
    "right",
  );
  expect_ascii_dist(new Translation(as_vec(-0.5, 0), box)).toMatchSnapshot(
    "left",
  );
  expect_ascii_dist(new Translation(as_vec(0, 0.5), box)).toMatchSnapshot(
    "top",
  );
  expect_ascii_dist(new Translation(as_vec(0, -0.5), box)).toMatchSnapshot(
    "bottom",
  );
});

test("rotate", () => {
  const box = new Translation(as_vec(0.25, 0), new Box(0.5, 0.3));
  expect_ascii_dist(new Rotation(angle_from_deg(0), box)).toMatchSnapshot("0");
  expect_ascii_dist(new Rotation(angle_from_deg(90), box)).toMatchSnapshot("0");
  expect_ascii_dist(new Rotation(angle_from_deg(45), box)).toMatchSnapshot("0");
});

test("scale", () => {
  const box = new Box(0.5, 0.5);
  expect_ascii_dist(new Scaling(as_num(2), box)).toMatchSnapshot("bigger");
  expect_ascii_dist(new Scaling(as_num(0.5), box)).toMatchSnapshot("smaller");
});

test("dilate", () => {
  const box = new Box(0.5, 0.3);
  expect_ascii_dist(new Dilatation(as_num(0.2), box)).toMatchSnapshot(
    "outside",
  );
  expect_ascii_dist(new Dilatation(as_num(-0.1), box)).toMatchSnapshot(
    "inside",
  );
});

test("shell", () => {
  const box = new Box(1.3, 0.7);
  expect_ascii_dist(new Shell(as_num(0.1), box)).toMatchSnapshot();
});

test("morph", () => {
  const box = new Box(1.8, 0.5);
  const circle = new Circle(0.5);

  expect_ascii_dist(new Morph(as_num(0.5), box, circle)).toMatchSnapshot(
    "halfway",
  );
  expect_ascii_dist(new Morph(as_num(0.9), box, circle)).toMatchSnapshot(
    "mostly circle",
  );
  expect_ascii_dist(new Morph(as_num(0.1), box, circle)).toMatchSnapshot(
    "mostly box",
  );
});

test("union", () => {
  const box = new Box(0.3, 1.9);
  const box2 = new Box(1.9, 0.3);
  const circle = new Circle(0.5);

  expect_ascii_dist(new Union(box, circle)).toMatchSnapshot();
  expect_ascii_dist(new Union(box, circle, box2)).toMatchSnapshot(
    "union of three",
  );
});

test("intersection", () => {
  const box = new Box(0.7, 1.9);
  const circle = new Circle(0.5);

  expect_ascii_dist(new Intersection(box, circle)).toMatchSnapshot();
  expect_ascii_dist(
    new Intersection(box, circle, new TopHalfPlane()),
  ).toMatchSnapshot("intersection of three");
});

test("difference", () => {
  const box = new Box(1.9, 0.7);
  const circle = new Circle(0.5);

  expect_ascii_dist(new Difference(circle, box)).toMatchSnapshot("remove box");
  expect_ascii_dist(new Difference(box, circle)).toMatchSnapshot(
    "remove circle",
  );
});
