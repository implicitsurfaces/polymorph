import { test } from "vitest";
import {
  Circle,
  Box,
  TopHalfPlane,
  BottomHalfPlane,
  LeftHalfPlane,
  RightHalfPlane,
} from "./profiles";

import { expect_ascii_dist } from "./test-utils";

test("circle", () => {
  expect_ascii_dist(new Circle(0.8)).toMatchSnapshot();
  expect_ascii_dist(new Circle(0.1)).toMatchSnapshot();
});

test("box", () => {
  expect_ascii_dist(new Box(1.7, 1)).toMatchSnapshot();
  expect_ascii_dist(new Box(0.2, 1.8)).toMatchSnapshot();
});

test("top half plane", () => {
  expect_ascii_dist(new TopHalfPlane()).toMatchSnapshot();
});

test("bottom half plane", () => {
  expect_ascii_dist(new BottomHalfPlane()).toMatchSnapshot();
});

test("left half plane", () => {
  expect_ascii_dist(new LeftHalfPlane()).toMatchSnapshot();
});

test("right half plane", () => {
  expect_ascii_dist(new RightHalfPlane()).toMatchSnapshot();
});
