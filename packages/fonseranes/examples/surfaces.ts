import {
  Cone,
  Curve3D,
  Cylinder,
  Point3D,
  circle,
  XY_Plane,
} from "../src/main";
import { outputSVG, sliceToHeight } from "./helpers";

const c2 = circle(0.3);
const c3 = new Curve3D(c2, XY_Plane.rotate(30, { x: -1, y: 1, z: 0 }));
const cyl = new Cylinder(c3);

outputSVG(sliceToHeight(cyl, 1, 10), "cylinder.svg");

const c3xy = new Curve3D(circle(0.3), XY_Plane);
const cone = new Cone(c3xy, new Point3D(0, 0.3, 1.001));

outputSVG(sliceToHeight(cone, 1, 10), "cone.svg");

const cone2 = new Cone(c3xy, new Point3D(0, 0, 1.001)).rotate(89, "x");

outputSVG(sliceToHeight(cone2, 1, 10), "cone-side.svg");
