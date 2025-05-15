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
  arc,
  YZ_Plane,
  circle,
  angleFromDeg,
  screenToScreenProjection,
  Vec2D,
  Vec3D,
} from "./src/main";
import { projectToPlaneInDirection } from "./src/transforms-helpers";
import { Transform3D } from "./src/transforms";
import fs from "node:fs";

import { outputSVG, sliceToHeight, movieSlices } from "./examples/helpers";

import { draw } from "./src/draw";

function extrude(curve: Curve3D, height: number) {
  const extrusion = new Cylinder(curve);
  const basePlane = curve.plane;
  const normal = basePlane.normal.normalize();
  const topPlane = basePlane.translate(normal.scale(height));
  return extrusion.partition(topPlane).partition(basePlane, true);
}

const baseTube = extrude(XY_Plane.embed(circle(0.5)), 0.8);
const cutPlane = YZ_Plane.rotate(40, "y").translateX(0.2);
const tube = baseTube.partition(cutPlane);

const slice = baseTube.slice(cutPlane);
const sideTube = new Cylinder(cutPlane.embed(slice))
  .partition(XY_Plane, true)
  .partition(cutPlane, true)
  .partition(YZ_Plane.translateX(0.7));

const planeOffset = 0;
let slicingPlane;
//slicingPlane = ZX_Plane.rotate(1e-10, "x").translateY(planeOffset);
slicingPlane = XY_Plane.translateZ(planeOffset);
//slicingPlane = YZ_Plane.rotate(1e-11, "y").translateX(planeOffset);

const shapeRotation = 0;
movieSlices([tube, sideTube], XY_Plane, [1e-10, 0.1, 0.2, 0.3, 0.4, 0.5]);

/**/
