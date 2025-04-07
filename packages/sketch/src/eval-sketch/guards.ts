import { Angle, Point, Vec2 } from "../geom";
import { Plane, Point3D, Vec3 } from "../geom-3d";
import { Num } from "../num";
import { DistField, Segment, SolidDistField } from "../types";
import { PartialPath } from "./handlers/utils/PartialPath";

export function guardPoint(value: unknown): Point {
  if (!(value instanceof Point)) {
    throw new Error(`Expected Point, got ${value}`);
  }
  return value;
}

export function guardNum(value: unknown): Num {
  if (!(value instanceof Num)) {
    throw new Error(`Expected number, got ${value}`);
  }
  return value;
}

export function guardAngle(value: unknown): Angle {
  if (!(value instanceof Angle)) {
    throw new Error(`Expected angle, got ${value}`);
  }
  return value;
}

export function guardVec2(value: unknown): Vec2 {
  if (!(value instanceof Vec2)) {
    throw new Error(`Expected Vec2, got ${value}`);
  }
  return value;
}

export function guardVec3(value: unknown): Vec3 {
  if (!(value instanceof Vec3)) {
    throw new Error(`Expected Vec3, got ${value}`);
  }
  return value;
}

export function guardPoint3(value: unknown): Point3D {
  if (!(value instanceof Point3D)) {
    throw new Error(`Expected Point3D, got ${value}`);
  }
  return value;
}

export function guardDistField(value: unknown): DistField {
  if (
    !value ||
    typeof value !== "object" ||
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    typeof (value as any).distanceTo !== "function"
  ) {
    throw new Error(`Expected DistField, got ${value}`);
  }

  return value as DistField;
}

export const guardProfile = guardDistField;

export function guardSolid(value: unknown): SolidDistField {
  if (
    !value ||
    typeof value !== "object" ||
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    typeof (value as any).valueAt !== "function"
  ) {
    throw new Error(`Expected SolidDistField, got ${value}`);
  }

  return value as SolidDistField;
}

export function guardSegmentCreator(
  value: unknown,
): (p0: Point, p1: Point) => Segment[] {
  if (typeof value !== "function") {
    throw new Error(`Expected function, got ${value}`);
  }
  return value as (p0: Point, p1: Point) => Segment[];
}

export function guardPartialPath(value: unknown): PartialPath {
  if (!(value instanceof PartialPath)) {
    throw new Error(`Expected PartialPath, got ${value}`);
  }
  return value;
}

export function guardPlane(value: unknown): Plane {
  if (!(value instanceof Plane)) {
    throw new Error(`Expected Plane, got ${value}`);
  }
  return value;
}

export function guardWidthModulation(value: unknown): (t: Num) => Num {
  if (typeof value !== "function") {
    throw new Error(`Expected function, got ${value}`);
  }
  return value as (t: Num) => Num;
}
