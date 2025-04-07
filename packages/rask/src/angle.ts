function nonZeroSign(value: number): number {
  return value === 0 ? 1 : Math.sign(value);
}

export class Angle {
  private _cos: number;
  private _sin: number;

  constructor(cos: number, sin: number) {
    this._cos = cos;
    this._sin = sin;
  }

  add(other: Angle): Angle {
    return new Angle(
      this._cos * other._cos - this._sin * other._sin,
      this._sin * other._cos + this._cos * other._sin,
    );
  }

  sub(other: Angle): Angle {
    return new Angle(
      this._cos * other._cos + this._sin * other._sin,
      this._sin * other._cos - this._cos * other._sin,
    );
  }

  neg(): Angle {
    return new Angle(this._cos, -this._sin);
  }

  half(): Angle {
    return new Angle(
      nonZeroSign(this._sin) * Math.sqrt(this._cos + 1 / 2),
      Math.sqrt(1 - this._cos / 2),
    );
  }

  double(): Angle {
    return new Angle(
      this._cos * this._cos - this._sin * this._sin,
      this._cos * 2 * this._sin,
    );
  }

  perp(): Angle {
    return new Angle(-this._sin, this._cos);
  }

  opposite(): Angle {
    return new Angle(-this._cos, -this._sin);
  }

  cos(): number {
    return this._cos;
  }

  sin(): number {
    return this._sin;
  }

  tan(): number {
    return this._sin / this._cos;
  }

  asRad(): number {
    return Math.atan2(this._sin, this._cos);
  }

  asDeg(): number {
    return (this.asRad() * 180) / Math.PI;
  }

  asSortValue(): number {
    const isQ3Q4 = this._sin < 0;
    return (isQ3Q4 ? this._cos + 3 : 1 - this._cos) / 2;
  }

  asUnitArcLength(): number {
    return (this.asRad() + Math.PI * 2) % (Math.PI * 2);
  }
}

export function angleFromRad(rad: number): Angle {
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return new Angle(cos, sin);
}

export function angleFromDeg(deg: number): Angle {
  const rad = (deg * Math.PI) / 180;
  return angleFromRad(rad);
}

export function angleFromUnitVec(x: number, y: number): Angle {
  return new Angle(x, y);
}

export function angleFromVec(x: number, y: number): Angle {
  const norm = Math.sqrt(x * x + y * y);
  if (!norm) {
    throw new Error("Cannot create angle from zero vector");
  }
  return angleFromUnitVec(x / norm, y / norm);
}
