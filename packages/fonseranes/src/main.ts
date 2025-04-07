import { circle, ellipse } from "./conic-sections";
import { Cone, Curve3D, Cylinder, Plane, Point3D } from "./primitives";

export * from "./primitives";
export * from "./angle";
export * from "./conic-sections";

export {
  Transform3D,
  Transform2D,
  rawTransform2D,
  rawTransform3D,
} from "./transforms";

export { renderAsSVG } from "./basic-2d-render";
export type { ShapeInput } from "./basic-2d-render";

export function cylinder(radius: number, secondaryRadius?: number) {
  const base = secondaryRadius
    ? ellipse(radius, secondaryRadius)
    : circle(radius);
  return new Cylinder(new Curve3D(base, XY_Plane));
}

export function straightCone(
  height: number,
  radius: number,
  secondaryRadius?: number,
) {
  const base = secondaryRadius
    ? ellipse(radius, secondaryRadius)
    : circle(radius);
  return new Cone(new Curve3D(base, XY_Plane), new Point3D(0, 0, height));
}

export const XY_Plane = new Plane(0, 0, 1, 0);
export const ZX_Plane = new Plane(0, -1, 0, 0);
export const YZ_Plane = new Plane(1, 0, 0, 0);
