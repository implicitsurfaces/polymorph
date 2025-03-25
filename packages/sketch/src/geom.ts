import { Num, ONE, asNum } from "./num";
import { atan2, hypot, ifTruthyElse } from "./num-ops";

function nonZeroSign(n: Num): Num {
  return ifTruthyElse(n.lessThan(0), asNum(-1), asNum(1));
}

const _2PI = asNum(Math.PI * 2);

export class Angle {
  private _cos: Num;
  private _sin: Num;

  constructor(cos: Num, sin: Num) {
    this._cos = cos;
    this._sin = sin;
  }

  add(other: Angle): Angle {
    return new Angle(
      this._cos.mul(other._cos).sub(this._sin.mul(other._sin)),
      this._sin.mul(other._cos).add(this._cos.mul(other._sin)),
    );
  }

  sub(other: Angle): Angle {
    return new Angle(
      this._cos.mul(other._cos).add(this._sin.mul(other._sin)),
      this._sin.mul(other._cos).sub(this._cos.mul(other._sin)),
    );
  }

  neg(): Angle {
    return new Angle(this._cos, this._sin.neg());
  }

  half(): Angle {
    return new Angle(
      nonZeroSign(this._sin).mul(this._cos.add(1).div(2).sqrt()),
      asNum(1).sub(this._cos).div(2).sqrt(),
    );
  }

  double(): Angle {
    return new Angle(
      this._cos.mul(this._cos).sub(this._sin.mul(this._sin)),
      this._cos.mul(2).mul(this._sin),
    );
  }

  perp(): Angle {
    return new Angle(this._sin.neg(), this._cos);
  }

  opposite(): Angle {
    return new Angle(this._cos.neg(), this._sin.neg());
  }

  cos(): Num {
    return this._cos;
  }

  sin(): Num {
    return this._sin;
  }

  tan(): Num {
    return this._sin.div(this._cos);
  }

  asRad(): Num {
    return atan2(this._sin, this._cos);
  }

  asDeg(): Num {
    return this.asRad().mul(180).div(Math.PI);
  }

  asSortValue(): Num {
    const isQ3Q4 = this._sin.lessThan(0);
    return ifTruthyElse(isQ3Q4, this._cos.add(3), asNum(1).sub(this._cos)).div(
      2,
    );
  }

  asUnitArcLength(): Num {
    return this.asRad().add(_2PI).mod(_2PI);
  }

  asVec(): UnitVec2 {
    return new UnitVec2(this._cos, this._sin);
  }
}

export function angleRromRad(rad: Num | number): Angle {
  const val = asNum(rad);
  return new Angle(val.cos(), val.sin());
}

export function angleFromDeg(deg: Num | number): Angle {
  const val = asNum(deg);
  return angleRromRad(val.mul(Math.PI).div(180));
}

export function angleFromSin(sin: Num | number): Angle {
  const val = asNum(sin);
  return new Angle(asNum(1).sub(val.square()).sqrt(), val);
}

export function angleFromCos(cos: Num | number): Angle {
  const val = asNum(cos);
  return new Angle(val, asNum(1).sub(val.square()).sqrt());
}

export function angleFromDirection(direction: Vec2): Angle {
  return direction.asAngle();
}

export function twoVectorsAngle(v1: Vec2, v2: Vec2): Angle {
  const u1 = v1.normalize();
  const u2 = v2.normalize();

  const cos = u1.dot(u2);
  const sin = u1.cross(u2);

  return new Angle(cos, sin);
}

export function arcTan(x: Num | number, y: Num | number): Angle {
  const norm = hypot(x, y);
  return new Angle(asNum(x).div(norm), asNum(y).div(norm));
}

export const NO_TURN = new Angle(asNum(1), asNum(0));
export const FULL_TURN = new Angle(asNum(1), asNum(0));
export const HALF_TURN = new Angle(asNum(-1), asNum(0));
export const QUARTER_TURN = new Angle(asNum(0), asNum(1));
export const THREE_QUARTER_TURN = new Angle(asNum(0), asNum(-1));
export const EIGHTH_TURN = new Angle(asNum(Math.SQRT1_2), asNum(Math.SQRT1_2));

export class Vec2 {
  protected _x: Num;
  protected _y: Num;

  constructor(x: Num, y: Num) {
    this._x = x;
    this._y = y;
  }

  add(other: Vec2): Vec2 {
    return new Vec2(this._x.add(other._x), this._y.add(other._y));
  }

  sub(other: Vec2): Vec2 {
    return new Vec2(this._x.sub(other._x), this._y.sub(other._y));
  }

  neg(): Vec2 {
    return new Vec2(this._x.neg(), this._y.neg());
  }

  scale(other: Num | number): Vec2 {
    return new Vec2(this._x.mul(other), this._y.mul(other));
  }

  div(other: Num | number): Vec2 {
    return new Vec2(this._x.div(other), this._y.div(other));
  }

  dot(other: Vec2): Num {
    return this._x.mul(other._x).add(this._y.mul(other._y));
  }

  cross(other: Vec2): Num {
    return this._x.mul(other._y).sub(this._y.mul(other._x));
  }

  norm(): Num {
    return this.dot(this).sqrt();
  }

  normalize(): UnitVec2 {
    return this.div(this.norm());
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }

  perp(): Vec2 {
    return new Vec2(this._y.neg(), this._x);
  }

  mirrorX(): Vec2 {
    return new Vec2(this._x.neg(), this._y);
  }

  mirrorY(): Vec2 {
    return new Vec2(this._x, this._y.neg());
  }

  rotate(angle: Angle): Vec2 {
    return new Vec2(
      angle.cos().mul(this._x).sub(angle.sin().mul(this._y)),
      angle.sin().mul(this._x).add(angle.cos().mul(this._y)),
    );
  }

  asAngle(): Angle {
    const normalized = this.normalize();
    return new Angle(normalized._x, normalized._y);
  }

  pointFromOrigin(): Point {
    return new Point(this._x, this._y);
  }
}

export class UnitVec2 extends Vec2 {
  asAngle(): Angle {
    return new Angle(this._x, this._y);
  }

  norm(): Num {
    return ONE;
  }

  normalize(): UnitVec2 {
    return this;
  }

  neg(): UnitVec2 {
    return new UnitVec2(this._x.neg(), this._y.neg());
  }

  perp(): UnitVec2 {
    return new UnitVec2(this._y.neg(), this._x);
  }

  mirrorX(): UnitVec2 {
    return new UnitVec2(this._x.neg(), this._y);
  }

  mirrorY(): UnitVec2 {
    return new UnitVec2(this._x, this._y.neg());
  }

  rotate(angle: Angle): UnitVec2 {
    return new UnitVec2(
      angle.cos().mul(this._x).sub(angle.sin().mul(this._y)),
      angle.sin().mul(this._x).add(angle.cos().mul(this._y)),
    );
  }
}

export class Point {
  constructor(
    private _x: Num,
    private _y: Num,
  ) {}

  add(vec: Vec2): Point {
    return new Point(this._x.add(vec.x), this._y.add(vec.y));
  }

  midPoint(other: Point): Point {
    return new Point(
      this._x.add(other._x).div(2),
      this._y.add(other._y).div(2),
    );
  }

  sub(vec: Vec2): Point {
    return new Point(this._x.sub(vec.x), this._y.sub(vec.y));
  }

  vecTo(other: Point): Vec2 {
    return new Vec2(other._x.sub(this._x), other._y.sub(this._y));
  }

  vecFrom(other: Point): Vec2 {
    return new Vec2(this._x.sub(other._x), this._y.sub(other._y));
  }

  vecFromOrigin(): Vec2 {
    return new Vec2(this._x, this._y);
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }
}

export const ORIGIN = new Point(asNum(0), asNum(0));

export function vecFromCartesianCoords(x: Num | number, y: Num | number): Vec2 {
  return new Vec2(asNum(x), asNum(y));
}
export const asVec = vecFromCartesianCoords;

export function vecFromPolarCoords(r: Num | number, angle: Angle): Vec2 {
  return new Vec2(angle.cos().mul(r), angle.sin().mul(r));
}

export class SolidAngle {
  private _turns: Num;

  constructor(turns: Num | number) {
    this._turns = asNum(turns);
  }

  get turns(): Num {
    return this._turns;
  }

  addAngle(angle: Angle): SolidAngle {
    return new SolidAngle(this._turns.add(angle.asRad().div(Math.PI * 2)));
  }

  addTurns(turns: Num | number): SolidAngle {
    return new SolidAngle(this._turns.add(asNum(turns)));
  }

  add(other: SolidAngle): SolidAngle {
    return new SolidAngle(this._turns.add(other._turns));
  }

  sub(other: SolidAngle): SolidAngle {
    return new SolidAngle(this._turns.sub(other._turns));
  }

  neg(): SolidAngle {
    return new SolidAngle(this._turns.neg());
  }

  half(): SolidAngle {
    return new SolidAngle(this._turns.div(2));
  }
}

export function solidAngleFromAngle(angle: Angle): SolidAngle {
  return new SolidAngle(0).addAngle(angle);
}

export function ifTruthyElseForAngles(
  condition: Num,
  ifTrue: Angle,
  ifFalse: Angle,
) {
  return new Angle(
    ifTruthyElse(condition, ifTrue.cos(), ifFalse.cos()),
    ifTruthyElse(condition, ifTrue.sin(), ifFalse.sin()),
  );
}

export function ifTruthyElseForPoints(
  condition: Num,
  ifTrue: Point,
  ifFalse: Point,
) {
  return new Point(
    ifTruthyElse(condition, ifTrue.x, ifFalse.x),
    ifTruthyElse(condition, ifTrue.y, ifFalse.y),
  );
}

export function ifTruthyElseForVec2s(
  condition: Num,
  ifTrue: Vec2,
  ifFalse: Vec2,
) {
  return new Vec2(
    ifTruthyElse(condition, ifTrue.x, ifFalse.x),
    ifTruthyElse(condition, ifTrue.y, ifFalse.y),
  );
}
