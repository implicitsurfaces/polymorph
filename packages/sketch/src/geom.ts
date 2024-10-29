import { Num, as_num } from "./num";
import { atan2, if_non_zero_else, less_than } from "./num-ops";

function non_zero_sign(n: Num): Num {
  return if_non_zero_else(less_than(n, 0), as_num(-1), as_num(1));
}

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
      non_zero_sign(this._sin).mul(this._cos.add(1).div(2).sqrt()),
      as_num(1).sub(this._cos).div(2).sqrt(),
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

  as_rad(): Num {
    return atan2(this._sin, this._cos);
  }

  as_deg(): Num {
    return this.as_rad().mul(180).div(Math.PI);
  }

  as_sort_value(): Num {
    const is_q2_q3 = less_than(this._sin, 0);
    return if_non_zero_else(
      is_q2_q3,
      this._cos.add(3),
      as_num(1).sub(this._cos),
    ).div(2);
  }

  as_vec(): Vec2 {
    return new Vec2(this._cos, this._sin);
  }
}

export function angle_from_rad(rad: Num | number): Angle {
  const val = as_num(rad);
  return new Angle(val.cos(), val.sin());
}

export function angle_from_deg(deg: Num | number): Angle {
  const val = as_num(deg);
  return angle_from_rad(val.mul(Math.PI).div(180));
}

export function angle_from_sin(sin: Num | number): Angle {
  const val = as_num(sin);
  return new Angle(as_num(1).sub(val.square()).sqrt(), val);
}

export function angle_from_cos(cos: Num | number): Angle {
  const val = as_num(cos);
  return new Angle(val, as_num(1).sub(val.square()).sqrt());
}

export function angle_from_direction(direction: Vec2): Angle {
  return direction.as_angle();
}

export const NO_TURN = new Angle(as_num(1), as_num(0));
export const FULL_TURN = new Angle(as_num(1), as_num(0));
export const HALF_TURN = new Angle(as_num(-1), as_num(0));
export const QUARTER_TURN = new Angle(as_num(0), as_num(1));
export const THREE_QUARTER_TURN = new Angle(as_num(0), as_num(-1));
export const EIGHTH_TURN = new Angle(
  as_num(Math.SQRT1_2),
  as_num(Math.SQRT1_2),
);

export class Vec2 {
  private _x: Num;
  private _y: Num;

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

  normalize(): Vec2 {
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

  as_angle(): Angle {
    const normalized = this.normalize();
    return new Angle(normalized._x, normalized._y);
  }

  point_from_origin(): Point {
    return new Point(this._x, this._y);
  }
}

export default class Point {
  constructor(
    private _x: Num,
    private _y: Num,
  ) {}

  add(vec: Vec2): Point {
    return new Point(this._x.add(vec.x), this._y.add(vec.y));
  }

  sub(vec: Vec2): Point {
    return new Point(this._x.sub(vec.x), this._y.sub(vec.y));
  }

  vec_to(other: Point): Vec2 {
    return new Vec2(other._x.sub(this._x), other._y.sub(this._y));
  }

  vec_from(other: Point): Vec2 {
    return new Vec2(this._x.sub(other._x), this._y.sub(other._y));
  }

  vec_from_origin(): Vec2 {
    return new Vec2(this._x, this._y);
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }
}

export const ORIGIN = new Point(as_num(0), as_num(0));

export function vec_from_cartesian_coords(
  x: Num | number,
  y: Num | number,
): Vec2 {
  return new Vec2(as_num(x), as_num(y));
}
export const as_vec = vec_from_cartesian_coords;

export function vec_from_polar_coords(r: Num | number, angle: Angle): Vec2 {
  return new Vec2(angle.cos().mul(r), angle.sin().mul(r));
}
