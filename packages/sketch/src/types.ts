import { Point, SolidAngle } from "./geom";
import { Point3D } from "./geom-3d";
import { Num } from "./num";
import { NumNode } from "./num-tree";

export interface DistField {
  distanceTo(point: Point): Num;
}

export interface Segment extends DistField {
  solidAngle(point: Point): SolidAngle;
}

export interface SolidDistField {
  valueAt(point: Point3D): Num;
}

export interface NumEvalKernel<T> {
  unaryOp(op: string, arg: T, node: NumNode): T;
  binaryOp(op: string, lhs: T, rhs: T, node: NumNode): T;
  variable(name: string, node: NumNode): T;
  literal(value: number, node: NumNode): T;
  value(value: T): number;
}
