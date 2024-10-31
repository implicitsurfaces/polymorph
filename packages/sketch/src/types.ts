import { Point, SolidAngle } from "./geom";
import { Num } from "./num";

export interface DistField {
  distanceTo(point: Point): Num;
}

export interface Segment extends DistField {
  solidAngle(point: Point): SolidAngle;
}
