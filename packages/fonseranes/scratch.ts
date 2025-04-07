import {
  Point3D,
  Curve3D,
  Cylinder,
  Cone,
  Patch,
  XY_Plane,
  cylinder,
  ZX_Plane,
  Point2D,
  Line2D,
} from "./src/main";
import fs from "node:fs";

import { outputSVG, sliceToHeight } from "./examples/helpers";

import { circle } from "./conic-sections";

const c = cylinder(0.3);

const p = c
  .clip(XY_Plane)
  .clip(XY_Plane.translateZ(0.2), true)
  .rotate(-30, "x");

outputSVG(sliceToHeight(p, 1, 30), "slice.svg", 0.1);
