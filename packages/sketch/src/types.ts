import Point from "./geom";
import { Num } from "./num";

export interface DistField {
  distanceTo(point: Point): Num;
}

export interface Segment {
  distanceTo(point: Point): Num;
  solidAngle(point: Point): Num;
}
