import { test } from "vitest";
import {
  LinearExtrusion2D,
  staticWidth,
  linearWidthVariation,
} from "./extrusions-2d";

import { expectASCIIDistance } from "./test-utils";
//import { asVec } from "./geom";
import { asNum } from "./num";

//const p = (x: number, y: number) => asVec(x, y).pointFromOrigin();

test("linear extrusion 2d", async () => {
  (
    await expectASCIIDistance(
      new LinearExtrusion2D(asNum(0.5), staticWidth(asNum(1))),
    )
  ).toMatchSnapshot();
});

test("linear extrusion 2d with linear width variation", async () => {
  (
    await expectASCIIDistance(
      new LinearExtrusion2D(
        asNum(0.8),
        linearWidthVariation(asNum(1.8), asNum(0.1)),
      ),
    )
  ).toMatchSnapshot();
});
