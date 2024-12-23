import { test } from "vitest";
import { LinearExtrusion2D, staticWidth } from "../extrusions-2d";

import { ex } from "../test-utils";
import { gradientAt, hypot } from "../num-ops";
import { asNum, Num, variable } from "../num";
import { asVec } from "../geom";

const p = (x: number | Num, y: number | Num) => asVec(x, y).pointFromOrigin();

test("interior grad of a linear extrusions", async () => {
  const extrusion = new LinearExtrusion2D(asNum(0.5), staticWidth(asNum(1)));
  const point = p(variable("x"), variable("y"));

  const grad = gradientAt(extrusion.distanceTo(point), [
    ["x", 0.01],
    ["y", 0.01],
  ]);

  ex(hypot(grad[0], grad[1])).toBeCloseTo(1);
});
