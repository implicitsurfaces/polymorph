import { test, describe } from "vitest";
import {
  LinearExtrusion2D,
  staticWidth,
  linearWidthVariation,
  OrientedLine,
  ArcExtrusion2D,
} from "./extrusions-2d";

import { ex, expectASCIIDistance } from "./test-utils";
import { asVec } from "./geom";
import { asNum } from "./num";
import { angleFromDeg } from "./geom";

const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

describe("linear extrusions", () => {
  test("standard", async () => {
    (
      await expectASCIIDistance(
        new LinearExtrusion2D(asNum(0.5), staticWidth(asNum(1))),
      )
    ).toMatchSnapshot();
  });

  test("with linear width variation", async () => {
    (
      await expectASCIIDistance(
        new LinearExtrusion2D(
          asNum(0.8),
          linearWidthVariation(asNum(1.8), asNum(0.1)),
        ),
      )
    ).toMatchSnapshot();
  });
});

test("oriented line", async () => {
  (await expectASCIIDistance(new OrientedLine(asNum(0.5)))).toMatchSnapshot();

  ex(new OrientedLine(asNum(0.5)).distanceTo(p(0.25, 1))).toBeCloseTo(1);
  ex(new OrientedLine(asNum(0.5)).distanceTo(p(0, -0.001))).toBeCloseTo(0.25);
  ex(new OrientedLine(asNum(0.5)).distanceTo(p(1, 0))).toBeCloseTo(0.75);

  ex(new OrientedLine(asNum(0.1)).distanceTo(p(0.05, 1))).toBeCloseTo(1);
  ex(new OrientedLine(asNum(0.1)).distanceTo(p(0, -0.001))).toBeCloseTo(0.05);
  ex(new OrientedLine(asNum(0.1)).distanceTo(p(1, 0))).toBeCloseTo(0.95);
});

describe("arc extrusions", () => {
  test("standard with angle smaller than 180", async () => {
    (
      await expectASCIIDistance(
        new ArcExtrusion2D(
          asNum(0.5),
          angleFromDeg(100),
          staticWidth(asNum(0.5)),
        ),
      )
    ).toMatchSnapshot();
  });

  test("with linear width variation", async () => {
    (
      await expectASCIIDistance(
        new ArcExtrusion2D(
          asNum(0.5),
          angleFromDeg(100),
          linearWidthVariation(asNum(0.6), asNum(0.1)),
        ),
      )
    ).toMatchSnapshot();
  });
});
