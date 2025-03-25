import { Angle, ifTruthyElseForPoints, Point, Vec2 } from "./geom";
import {
  closestPointOnEllipse,
  ellipseQuarticFactors,
  hyperbolaQuarticFactors,
} from "./geom-utils/closestPointOnEllipse";
import { Matrix2x2, Matrix3x3, RowVec3 } from "./geom-utils/matrices";
import { randInit } from "./geom-utils/pseudo-random";
import { solveQuartic } from "./geom-utils/solve-quartic";
import { asNum, NEG_ONE, Num, ONE, TWO, ZERO } from "./num";
import { ifTruthyElse, min } from "./num-ops";
import {
  rawTransform,
  rotationTransform,
  scalingTransform,
  Transform,
  translationTransform,
} from "./transforms-2d";

const Q_MAT = new Matrix3x3(
  ONE,
  ZERO,
  ZERO,
  ZERO,
  ONE,
  ZERO,
  ZERO,
  ZERO,
  NEG_ONE,
);

export class ConicProfile {
  private readonly _matrix: Matrix3x3;
  private readonly _subMatrix: Matrix2x2;

  private pointTransformCache: Transform | null = null;

  constructor(public readonly transformation: Transform) {
    this._matrix = transformation.matrix
      .transpose()
      .mul(Q_MAT)
      .mul(transformation.matrix);

    this._subMatrix = new Matrix2x2(
      this._matrix.x11,
      this._matrix.x12,
      this._matrix.x21,
      this._matrix.x22,
    );
  }

  transform(transform: Transform): ConicProfile {
    return new ConicProfile(this.transformation.precededBy(transform));
  }

  get pointTransform() {
    if (this.pointTransformCache) {
      return this.pointTransformCache;
    }

    this.pointTransformCache = translationTransform(
      this.center.vecFromOrigin().neg(),
    ).followedBy(rotationTransform(this.tilt.neg()));

    return this.pointTransformCache;
  }

  get center(): Point {
    const a = this._matrix.x11;
    const b = this._matrix.x12.mul(2);
    const c = this._matrix.x22;
    const d = this._matrix.x13.mul(2);
    const e = this._matrix.x23.mul(2);

    const denom = a.mul(c).mul(4).sub(b.square());

    const x = b.mul(e).sub(c.mul(d).mul(TWO)).div(denom);
    const y = b.mul(d).sub(a.mul(e).mul(TWO)).div(denom);

    return new Point(x, y);
  }

  get isCircle(): Num {
    return this._subMatrix.det().equals(ONE);
  }

  get isEllipse(): Num {
    return this._subMatrix.det().greaterThan(ZERO);
  }

  get radiuses(): [Num, Num] {
    const [r1, r2] = this._subMatrix.eigenvalues();
    return [ONE.div(r1.abs().sqrt()), ONE.div(r2.abs().sqrt())];
  }

  get tilt(): Angle {
    const eigenvalues = this._subMatrix.eigenvalues();
    const direction = this._subMatrix.eigenvector(eigenvalues[0]);

    return new Vec2(direction.x1, direction.x2).asAngle();
  }

  private candidatePoints(point: Point): Point[] {
    const a = this._matrix.x11;
    const b = this._matrix.x12;
    const c = this._matrix.x22;
    const d = this._matrix.x13;
    const e = this._matrix.x23;
    const f = this._matrix.x33;

    const x0 = point.x;
    const y0 = point.y;

    const _2neg = asNum(-2);
    const _4 = asNum(4);
    const _4neg = asNum(-4);

    const l0 = f
      .add(a.mul(x0.square()))
      .add(c.mul(y0.square()))
      .add(TWO.mul(d).mul(x0))
      .add(TWO.mul(e).mul(y0))
      .add(TWO.mul(b).mul(x0).mul(y0));

    const l1 = _2neg
      .mul(d.square())
      .add(_2neg.mul(e.square()))
      .add(_2neg.mul(b.square()).mul(x0.square()))
      .add(_2neg.mul(b.square()).mul(y0.square()))
      .add(TWO.mul(a).mul(f))
      .add(TWO.mul(c).mul(f))
      .add(_4neg.mul(b).mul(d).mul(y0))
      .add(_4neg.mul(b).mul(e).mul(x0))
      .add(TWO.mul(a).mul(c).mul(x0.square()))
      .add(TWO.mul(a).mul(c).mul(y0.square()))
      .add(_4.mul(a).mul(e).mul(y0))
      .add(_4.mul(c).mul(d).mul(x0));

    const l2 = f
      .mul(a.square())
      .add(f.mul(c.square()))
      .add(NEG_ONE.mul(a).mul(d.square()))
      .add(NEG_ONE.mul(c).mul(e.square()))
      .add(_4neg.mul(a).mul(e.square()))
      .add(_4neg.mul(c).mul(d.square()))
      .add(_2neg.mul(f).mul(b.square()))
      .add(a.mul(c.square()).mul(x0.square()))
      .add(c.mul(a.square()).mul(y0.square()))
      .add(NEG_ONE.mul(a).mul(b.square()).mul(y0.square()))
      .add(NEG_ONE.mul(c).mul(b.square()).mul(x0.square()))
      .add(TWO.mul(d).mul(x0).mul(b.square()))
      .add(TWO.mul(d).mul(x0).mul(c.square()))
      .add(TWO.mul(e).mul(y0).mul(a.square()))
      .add(TWO.mul(e).mul(y0).mul(b.square()))
      .add(TWO.mul(x0).mul(y0).mul(b.square().mul(b)))
      .add(_4.mul(a).mul(c).mul(f))
      .add(asNum(6.0).mul(b).mul(d).mul(e))
      .add(_2neg.mul(a).mul(b).mul(d).mul(y0))
      .add(_2neg.mul(a).mul(b).mul(e).mul(x0))
      .add(_2neg.mul(b).mul(c).mul(d).mul(y0))
      .add(_2neg.mul(b).mul(c).mul(e).mul(x0))
      .add(_2neg.mul(a).mul(b).mul(c).mul(x0).mul(y0));

    const l3 = _2neg
      .mul(a.square())
      .mul(e.square())
      .add(_2neg.mul(c.square()).mul(d.square()))
      .add(_2neg.mul(a).mul(c).mul(d.square()))
      .add(_2neg.mul(a).mul(c).mul(e.square()))
      .add(_2neg.mul(a).mul(f).mul(b.square()))
      .add(_2neg.mul(c).mul(f).mul(b.square()))
      .add(TWO.mul(a).mul(f).mul(c.square()))
      .add(TWO.mul(c).mul(f).mul(a.square()))
      .add(_4.mul(a).mul(b).mul(d).mul(e))
      .add(_4.mul(b).mul(c).mul(d).mul(e));

    const l4 = f
      .mul(b.square().square())
      .add(
        a
          .mul(e.square())
          .add(c.mul(d.square()))
          .sub(a.mul(c).mul(f).mul(TWO))
          .sub(d.mul(e).mul(b).mul(TWO))
          .mul(b.square()),
      )
      .add(f.mul(a.square()).mul(c.square()))
      .sub(a.mul(c.square()).mul(d.square()))
      .sub(c.mul(a.square()).mul(e.square()))
      .add(TWO.mul(a).mul(b).mul(c).mul(d).mul(e));

    const lamba = solveQuartic(l4, l3, l2, l1, l0);

    const aPlusC = a.add(c);
    const acMinusBSquared = a.mul(c).sub(b.square());

    const ldet = a.sub(c).square().add(b.mul(b).mul(4)).sqrt();
    const d0 = aPlusC.add(ldet).div(2).div(acMinusBSquared).neg();
    const d1 = aPlusC.sub(ldet).div(2).div(acMinusBSquared).neg();

    const rx0b = c.mul(x0).sub(d.add(b.mul(y0)));
    const rx1b = b.mul(e).sub(c.mul(d));

    const ry0b = a.mul(y0).sub(e.add(b.mul(x0)));
    const ry1b = b.mul(d).sub(a.mul(e));

    return lamba.map((l) => {
      const ld0 = l.sub(d0);
      const ld1 = l.sub(d1);
      const denom = ld0.mul(ld1).mul(acMinusBSquared);

      const validX = x0.add(l.mul(rx0b.add(l.mul(rx1b)))).div(denom);
      const validY = y0.add(l.mul(ry0b.add(l.mul(ry1b)))).div(denom);

      const invalidSolution = ld0
        .abs()
        .lessThan(1e-15)
        .or(ld1.abs().lessThan(1e-15));

      const x = ifTruthyElse(invalidSolution, asNum(1e100), validX);
      const y = ifTruthyElse(invalidSolution, 1e100, validY);

      return new Point(x, y);
    });
  }

  occupancySign(point: Point): Num {
    const r = new RowVec3(point.x, point.y, ONE);
    const c = r.transpose();

    const v = r.product(this._matrix).dot(c);
    return v.sign();
  }

  closestPoint(point: Point): Point {
    const candidates = this.candidatePoints(point);

    let closest = candidates[0];
    let minDistance = closest.vecFrom(point).norm();

    for (let i = 1; i < candidates.length; i++) {
      const distance = candidates[i].vecFrom(point).norm();

      closest = ifTruthyElseForPoints(
        distance.lessThan(minDistance),
        candidates[i],
        closest,
      );
      minDistance = min(minDistance, distance);
    }

    return closest;
  }

  distanceToEllipse(point: Point): Num {
    const newPoint = this.pointTransform.apply(point);
    const [r1, r2] = this.radiuses;

    const distance = closestPointOnEllipse(r1, r2, newPoint)
      .vecFrom(newPoint)
      .norm();

    return this.occupancySign(point).mul(distance);
  }

  distanceToGeneric(point: Point): Num {
    const candidates = this.candidatePoints(point);

    const distances = candidates.map((candidate) =>
      candidate.vecFrom(point).norm(),
    );

    const minDistance = min(...(distances as [Num]));

    return minDistance.mul(this.occupancySign(point));
  }

  distanceToRegular(point: Point): Num {
    const newPoint = this.pointTransform.apply(point);

    const [r1, r2] = this.radiuses;

    const ellipseFactors = ellipseQuarticFactors(r1, r2, newPoint);
    const hyperbolaFactors = hyperbolaQuarticFactors(r1, r2, newPoint);

    const isEllipse = this.isEllipse;

    const factors = ellipseFactors.map((factor, i) =>
      ifTruthyElse(isEllipse, factor, hyperbolaFactors[i]),
    );

    const params = solveQuartic(...(factors as [Num, Num, Num, Num, Num]));

    const points = params.flatMap((param) => {
      const sinParam = ONE.sub(param.square()).sqrt();

      const x = param.mul(r1);
      const y = sinParam.mul(r2);

      return [new Point(x, y), new Point(x, y.neg())];
    });

    const distances = points.map((candidate) =>
      candidate.vecFrom(newPoint).norm(),
    );

    const minDistance = min(...(distances as [Num]));
    return this.occupancySign(point).mul(minDistance);
  }

  distanceTo(point: Point): Num {
    return this.distanceToGeneric(point);
  }
}

const hyperbolaProjection = (minorRadius: Num) =>
  rawTransform(
    ONE,
    ZERO,
    ZERO,
    ZERO,
    ZERO,
    ONE.div(minorRadius),
    ZERO,
    minorRadius,
    ZERO,
  );

export function circleConic(radius: Num): ConicProfile {
  return new ConicProfile(scalingTransform(radius, radius).reverse());
}

export function hyperbolaConic(
  majorRadius: Num,
  minorRadius: Num,
): ConicProfile {
  return new ConicProfile(
    scalingTransform(majorRadius, minorRadius)
      .reverse()
      .compose(hyperbolaProjection(minorRadius).reverse()),
  );
}

export function ellipseConic(majorRadius: Num, minorRadius: Num): ConicProfile {
  return new ConicProfile(
    scalingTransform(majorRadius.inv(), minorRadius.inv()),
  );
}

export function genericEllipseConic(
  majorRadius: Num,
  minorRadius: Num,
  rotationAngle: Angle,
  center: Point,
) {
  return new ConicProfile(
    scalingTransform(majorRadius, minorRadius)
      .reverse()
      .compose(rotationTransform(rotationAngle.neg()))
      .compose(translationTransform(center.vecFromOrigin().neg())),
  );
}

export function randomConic(seed: Num | number) {
  const rand = randInit(asNum(seed));
  const trans = rawTransform(
    rand(),
    rand(),
    rand(),
    rand(),
    rand(),
    rand(),
    rand(),
    rand(),
    rand(),
  );

  return new ConicProfile(trans);
}
