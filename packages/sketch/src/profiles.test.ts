import { test } from "vitest";
import {
  Circle,
  Box,
  TopHalfPlane,
  BottomHalfPlane,
  LeftHalfPlane,
  RightHalfPlane,
} from "./profiles";

import { expectASCIIshape, expectASCIIDistance } from "./test-utils";

test("circle", () => {
  expectASCIIshape(new Circle(0.8)).toMatchSnapshot();
  expectASCIIshape(new Circle(0.1)).toMatchSnapshot();
  expectASCIIDistance(new Circle(0.8)).toMatchSnapshot();
});

test("box", () => {
  expectASCIIshape(new Box(1.7, 1)).toMatchSnapshot();
  expectASCIIshape(new Box(0.2, 1.8)).toMatchSnapshot();
});

test("top half plane", () => {
  expectASCIIshape(new TopHalfPlane()).toMatchSnapshot();
});

test("bottom half plane", () => {
  expectASCIIshape(new BottomHalfPlane()).toMatchSnapshot();
});

test("left half plane", () => {
  expectASCIIshape(new LeftHalfPlane()).toMatchSnapshot();
});

test("right half plane", () => {
  expectASCIIshape(new RightHalfPlane()).toMatchSnapshot();
});
