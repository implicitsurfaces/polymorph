import { Angle, Point, Vec2 } from "./geom";
import { Num, as_num } from "./num";
import { max, min } from "./num-ops";
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
    const rotated = point.vec_from_origin().rotate(this.angle.neg());
    return this.sdf.distanceTo(rotated.point_from_origin());
  }
}

export class Scaling implements DistField {
  constructor(
    readonly factor: Num,
    readonly sdf: DistField,
  ) {}

  distanceTo(point: Point): Num {
    return this.sdf
      .distanceTo(point.vec_from_origin().div(this.factor).point_from_origin())
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
    const scaled1 = as_num(1).sub(this.t).mul(this.sdf1.distanceTo(point));
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
