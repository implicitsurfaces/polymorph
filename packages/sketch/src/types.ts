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

export type UnaryOperation =
  | "SQRT"
  | "CBRT"
  | "COS"
  | "ACOS"
  | "ASIN"
  | "TAN"
  | "ATAN"
  | "LOG"
  | "EXP"
  | "ABS"
  | "NEG"
  | "SIN"
  | "SIGN"
  | "NOT"
  | "TANH"
  | "LOG1P"
  | "DEBUG";

export type BinaryOperation =
  | "ADD"
  | "SUB"
  | "MUL"
  | "DIV"
  | "MOD"
  | "ATAN2"
  | "MIN"
  | "MAX"
  | "COMPARE"
  | "AND"
  | "OR";

export interface NumEvalKernel<T> {
  unaryOp(op: UnaryOperation, arg: T, node: NumNode): T;
  binaryOp(op: BinaryOperation, lhs: T, rhs: T, node: NumNode): T;
  variable(name: string, node: NumNode): T;
  literal(value: number, node: NumNode): T;
  value(value: T): number;
}
