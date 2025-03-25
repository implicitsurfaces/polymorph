import { test } from "vitest";
import { ellipseConic } from "./conic";
import { asVec3, XY_PLANE, XZ_PLANE } from "./geom-3d";
import { SolidSlice } from "./profiles";
import { ConicFrustum } from "./solids";
import { expectASCIIDistance } from "./test-utils";
import { asNum } from "./num";

test("frustum from an ellipse conic", async () => {
  const ellipse = ellipseConic(asNum(0.8), asNum(0.2));
  const frustum = new ConicFrustum(ellipse, asNum(1));

  const p = XZ_PLANE.translate(asVec3(0, 0, 0.1));
  (await expectASCIIDistance(new SolidSlice(frustum, p))).toMatchSnapshot();

  const p2 = XY_PLANE.translate(asVec3(0, 0, 1));
  (await expectASCIIDistance(new SolidSlice(frustum, p2))).toMatchSnapshot();
});
