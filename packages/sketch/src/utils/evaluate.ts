import { Angle, Point, Vec2 } from "../geom";
import {
  Matrix2x2,
  Matrix3x3,
  ColVec2,
  ColVec3,
  RowVec2,
  RowVec3,
} from "../geom-utils/matrices";
import { Num } from "../num";
import { simpleEval } from "../eval-num/js-eval";
import { NumNode } from "../num-tree";

type BasicTypes =
  | Num
  | Vec2
  | Point
  | Angle
  | Matrix3x3
  | Matrix2x2
  | ColVec2
  | RowVec2
  | ColVec3
  | RowVec3;

function evaluate(num: Num | Angle, logDebug?: boolean): number;
function evaluate(
  num: Exclude<BasicTypes, Num | Angle> | Num[] | Angle[],
  logDebug?: boolean,
): number[];
function evaluate(
  num: BasicTypes | Num[] | Angle[],
  logDebug = false,
): number | number[] {
  const ev = (n: NumNode) => simpleEval(n, new Map(), logDebug);

  if (Array.isArray(num)) {
    return num.map((n) => evaluate(n));
  } else if (num instanceof Num) {
    return ev(num.n);
  } else if (num instanceof Point) {
    return [ev(num.x.n), ev(num.y.n)];
  } else if (num instanceof Vec2) {
    return [ev(num.x.n), ev(num.y.n)];
  } else if (num instanceof Angle) {
    return ev(num.asDeg().n);
  } else if (num instanceof Matrix2x2) {
    return [ev(num.x11.n), ev(num.x12.n), ev(num.x21.n), ev(num.x22.n)];
  } else if (num instanceof Matrix3x3) {
    return [
      ev(num.x11.n),
      ev(num.x12.n),
      ev(num.x13.n),
      ev(num.x21.n),
      ev(num.x22.n),
      ev(num.x23.n),
      ev(num.x31.n),
      ev(num.x32.n),
      ev(num.x33.n),
    ];
  } else if (num instanceof ColVec2 || num instanceof RowVec2) {
    return [ev(num.x1.n), ev(num.x2.n)];
  } else if (num instanceof ColVec3 || num instanceof RowVec3) {
    return [ev(num.x1.n), ev(num.x2.n), ev(num.x3.n)];
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  throw new Error(`Unknown type ${(num as any).constructor.name}`);
}

export { evaluate };
