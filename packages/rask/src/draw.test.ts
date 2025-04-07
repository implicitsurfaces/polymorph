import {
  Conic2D,
  Conic3D,
  Line2D,
  Line3D,
  Plane,
  PlanarCoordinateSystem,
  Point2D,
  Segment2D,
  Vec2D,
  Point3D,
  Curve3D,
  Cylinder,
  Cone,
} from "./primitives";
import fs from "node:fs";

import {
  rawTransform2D,
  rotate,
  rotate3D,
  rotationAroundXAxisTransform3D,
  rotationAroundYAxisTransform3D,
  rotationAroundZAxisTransform3D,
  rotationTransform3D,
  scale,
  scalingTransform2D,
  translate,
  translate3D,
  translationTransform3D,
} from "./transforms";

import { renderAsSVG, ShapeInput } from "./basic-2d-render";
import { arc, circle, ellipse, hyperbola, parabola } from "./conic-sections";
import { Angle, angleFromDeg } from "./angle";
import { Matrix3x4, Matrix4x4 } from "./matrices";
import { canonicalCoordinateSystem } from "./plane-helpers";

type Shape2D = Line2D | Conic2D | Point2D | Segment2D;

const outputSVG = (
  shape: ShapeInput | ShapeInput[],
  filename = "test.svg",
  threshold = 0,
) => {
  const svg = renderAsSVG(shape, { threshold });
  fs.writeFileSync(filename, svg);
};

const p0 = new Point2D(-0.8, 0.2);
const p1 = new Point2D(0.5, 0.5);

outputSVG(
  [scale(p0, 1, 1.6), scale(p1, 1, 1.6), scale(arc(p0, p1, 0.6), 1, 1.6)],
  "arc.svg",
  0.1,
);

const XY_Plane = new Plane(0, 0, 1, 0);
const ZX_Plane = new Plane(0, -1, 0, 0);
const YZ_Plane = new Plane(1, 0, 0, 0);

console.log("YZ_Plane", XY_Plane);

const translation = translationTransform3D(0.5, 0, 1);
const rotationZ = rotationAroundZAxisTransform3D(angleFromDeg(45));
const rotationX = rotationAroundXAxisTransform3D(angleFromDeg(45));

const pl = translate3D(XY_Plane, { x: 0.5, y: 0, z: 1 });
const trans = rotationZ;

function axesProjection(shape: Line3D | Conic3D, filename: string = "test") {
  const xTransform = shape.projectIntoPlane(XY_Plane);
  const yTransform = shape.projectIntoPlane(ZX_Plane);
  const zTransform = shape.projectIntoPlane(YZ_Plane);

  fs.writeFileSync(`${filename}-xy.svg`, renderAsSVG(xTransform.curve));
  fs.writeFileSync(`${filename}-zx.svg`, renderAsSVG(yTransform.curve));
  fs.writeFileSync(`${filename}-yz.svg`, renderAsSVG(zTransform.curve));
}

const c2 = circle(0.3);
const c3 = new Curve3D(
  c2,
  XY_Plane.transform(rotationTransform3D(angleFromDeg(30), 1, 0, 0)),
);

const cyl = new Cylinder(c3);

const c3xy = new Curve3D(ellipse(0.2, 0.3), XY_Plane);
let cone = new Cone(c3xy, new Point3D(0, 0, 1.001));
cone = translate3D(cone, { x: 0.1, y: 0, z: 0 });

//cone = rotate3D(cone, angleFromDeg(30));

function sliceToHeight(
  shape: any,
  height: number,
  steps: number,
  basePlane = XY_Plane,
) {
  const step = height / steps;
  const slices: ShapeInput[] = [shape.slice(basePlane).curve];
  for (let i = step; i < height; i += step) {
    slices.push({
      shape: shape.slice(translate3D(basePlane, { x: 0, y: 0, z: i })).curve,
      strokeWidth: 0.4,
    });
  }

  slices.push(
    shape.slice(translate3D(basePlane, { x: 0, y: 0, z: height })).curve,
  );

  return slices;
}

outputSVG(sliceToHeight(cyl, 1, 10), "cylinder.svg");
outputSVG(sliceToHeight(cone, 1, 10), "cone.svg");
