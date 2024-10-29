import Point from "./geom";
import { Num } from "./num";

export interface DistField {
  distance_to(point: Point): Num;
}
