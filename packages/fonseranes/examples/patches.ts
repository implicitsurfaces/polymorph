import { cylinder, straightCone, ZX_Plane } from "../src/main";
import { outputSVG, sliceToHeight } from "./helpers";

const patch = cylinder(0.6, 0.3)
  .clip(ZX_Plane.rotate(30, "z"))
  .rotate(-30, "x");

outputSVG(sliceToHeight(patch, 1, 10), "patch-cylinder.svg", 0.01, 0.1);

const patch2 = straightCone(1.2, 0.3, 0.2)
  .clip(ZX_Plane.rotate(30, "z").rotate(10, "x"))
  .rotate(-30, "x");

outputSVG(sliceToHeight(patch2, 1, 10), "patch-cone.svg", 0.01, 0.1);
