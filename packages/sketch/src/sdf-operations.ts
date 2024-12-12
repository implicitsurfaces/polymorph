import { Angle, Point, Vec2 } from "./geom";
import { Num, ONE, asNum } from "./num";
import { max, min, clamp } from "./num-ops";
import { DistField } from "./types";

export class Translation implements DistField {
  constructor(
    readonly offset: Vec2,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return this.sdf.distanceTo(point.sub(this.offset));
  }
}

export class Rotation implements DistField {
  constructor(
    readonly angle: Angle,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const rotated = point.vecFromOrigin().rotate(this.angle.neg());
    return this.sdf.distanceTo(rotated.pointFromOrigin());
  }
}

export class Flip implements DistField {
  constructor(
    readonly axis: "x" | "y",
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const v = point.vecFromOrigin();
    const flipped = this.axis === "x" ? v.mirrorX() : v.mirrorY();
    return this.sdf.distanceTo(flipped.pointFromOrigin());
  }
}

export class Scaling implements DistField {
  constructor(
    readonly factor: Num,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return this.sdf
      .distanceTo(point.vecFromOrigin().div(this.factor).pointFromOrigin())
      .mul(this.factor);
  }
}

export class Dilatation implements DistField {
  constructor(
    readonly offset: Num,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return this.sdf.distanceTo(point).sub(this.offset);
  }
}

export class Shell implements DistField {
  constructor(
    readonly thickness: Num,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return this.sdf.distanceTo(point).abs().sub(this.thickness);
  }
}

export class Morph implements DistField {
  constructor(
    readonly t: Num,
    readonly sdf1: DistField,
    readonly sdf2: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const scaled1 = asNum(1).sub(this.t).mul(this.sdf1.distanceTo(point));
    const scaled2 = this.t.mul(this.sdf2.distanceTo(point));
    return scaled1.add(scaled2);
  }
}

export class Union implements DistField {
  readonly first: DistField;
  readonly others: DistField[];

  constructor(first: DistField, ...others: DistField[]) {
    this.first = first;
    this.others = others;
  }

  distanceTo(point: Point): Num {
    return min(
      this.first.distanceTo(point),
      ...this.others.map((sdf) => sdf.distanceTo(point)),
    );
  }
}

export class SmoothUnion implements DistField {
  constructor(
    readonly k: Num,
    readonly first: DistField,
    readonly second: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const d1 = this.first.distanceTo(point);
    const d2 = this.second.distanceTo(point);

    const h = clamp(d2.sub(d1).div(this.k).mul(0.5).add(0.5), 0, 1);
    return d2.add(d1.sub(d2).mul(h)).sub(this.k.mul(h).mul(ONE.sub(h)));
  }
}

export class Intersection implements DistField {
  readonly first: DistField;
  readonly others: DistField[];

  constructor(first: DistField, ...others: DistField[]) {
    this.first = first;
    this.others = others;
  }

  distanceTo(point: Point): Num {
    return max(
      this.first.distanceTo(point),
      ...this.others.map((sdf) => sdf.distanceTo(point)),
    );
  }
}

function smoothIntersect(k: Num, d1: Num, d2: Num) {
  const h = clamp(d2.sub(d1).div(k).mul(0.5).neg().add(0.5), 0, 1);
  return d2.add(d1.sub(d2).mul(h)).sub(k.mul(h).mul(asNum(1).sub(h)));
}

export class SmoothIntersection implements DistField {
  constructor(
    readonly k: Num,
    readonly first: DistField,
    readonly second: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const d1 = this.first.distanceTo(point);
    const d2 = this.second.distanceTo(point);

    return smoothIntersect(this.k, d1, d2);
  }
}

export class Difference implements DistField {
  constructor(
    readonly first: DistField,
    readonly other: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return max(
      this.first.distanceTo(point),
      this.other.distanceTo(point).neg(),
    );
  }
}

export class SmoothDifference implements DistField {
  constructor(
    readonly k: Num,
    readonly first: DistField,
    readonly second: DistField,
  ) {}

  distanceTo(point: Point): Num {
    const d1 = this.first.distanceTo(point);
    const d2 = this.second.distanceTo(point);

    return smoothIntersect(this.k, d1, d2.neg());
  }
}
