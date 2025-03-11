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

function evaluate(num: Num | Angle): number;
function evaluate(
  num: Exclude<BasicTypes, Num | Angle> | Num[] | Angle[],
): number[];
function evaluate(num: BasicTypes | Num[] | Angle[]): number | number[] {
  if (Array.isArray(num)) {
    return num.map((n) => evaluate(n));
  } else if (num instanceof Num) {
    return simpleEval(num.n);
  } else if (num instanceof Point) {
    return [simpleEval(num.x.n), simpleEval(num.y.n)];
  } else if (num instanceof Vec2) {
    return [simpleEval(num.x.n), simpleEval(num.y.n)];
  } else if (num instanceof Angle) {
    return simpleEval(num.asDeg().n);
  } else if (num instanceof Matrix2x2) {
    return [
      simpleEval(num.x11.n),
      simpleEval(num.x12.n),
      simpleEval(num.x21.n),
      simpleEval(num.x22.n),
    ];
  } else if (num instanceof Matrix3x3) {
    return [
      simpleEval(num.x11.n),
      simpleEval(num.x12.n),
      simpleEval(num.x13.n),
      simpleEval(num.x21.n),
      simpleEval(num.x22.n),
      simpleEval(num.x23.n),
      simpleEval(num.x31.n),
      simpleEval(num.x32.n),
      simpleEval(num.x33.n),
    ];
  } else if (num instanceof ColVec2 || num instanceof RowVec2) {
    return [simpleEval(num.x1.n), simpleEval(num.x2.n)];
  } else if (num instanceof ColVec3 || num instanceof RowVec3) {
    return [simpleEval(num.x1.n), simpleEval(num.x2.n), simpleEval(num.x3.n)];
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  throw new Error(`Unknown type ${(num as any).constructor.name}`);
}

export { evaluate };
