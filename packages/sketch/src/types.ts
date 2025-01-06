import { Point, SolidAngle } from "./geom";
import { Point3D } from "./geom-3d";
import { Num } from "./num";

export interface DistField {
  distanceTo(point: Point): Num;
}

export interface Segment extends DistField {
  solidAngle(point: Point): SolidAngle;
}

export interface SolidDistField {
  valueAt(point: Point3D): Num;
}
