import { Point } from "./geom";
import { Matrix3x3, RowVec3 } from "./geom-utils/matrices";
import { solveQuartic } from "./geom-utils/solve-polynomial";
import { asNum, NEG_ONE, Num, ONE, TWO, ZERO } from "./num";
import { min } from "./num-ops";
import { scalingTransform, Transform } from "./transforms-2d";

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

export class Conic {
  private readonly _matrix: Matrix3x3;

  constructor(public readonly transformation: Transform) {
    this._matrix = transformation.matrix
      .transpose()
      .mul(Q_MAT)
      .mul(transformation.matrix);
  }

  private candidatePoints(point: Point): Point[] {
    const a = this._matrix.x11;
    const b = this._matrix.x12.mul(4);
    const c = this._matrix.x22;
    const d = this._matrix.x13.mul(4);
    const e = this._matrix.x23.mul(4);
    const f = this._matrix.x33;

    const x0 = point.x;
    const y0 = point.y;

    const _2neg = asNum(-2);
    const _4 = asNum(4);
    const _4neg = asNum(-4);

    const l4 = f
      .add(a.mul(x0.square()))
      .add(c.mul(y0.square()))
      .add(TWO.mul(d).mul(x0))
      .add(TWO.mul(e).mul(y0))
      .add(TWO.mul(b).mul(x0).mul(y0));

    const l3 = _2neg
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

    const l1 = _2neg
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

    const l0 = f
      .mul(b.square().square())
      .add(a.mul(b.square()).mul(e.square()))
      .add(c.mul(b.square()).mul(d.square()))
      .add(f.mul(a.square()).mul(c.square()))
      .add(NEG_ONE.mul(a).mul(c.square()).mul(d.square()))
      .add(NEG_ONE.mul(c).mul(a.square()).mul(e.square()))
      .add(_2neg.mul(d).mul(e).mul(b.square().mul(b)))
      .add(_2neg.mul(a).mul(c).mul(f).mul(b.square()))
      .add(TWO.mul(a).mul(b).mul(c).mul(d).mul(e));

    const lamba = solveQuartic(l4, l3, l2, l1, l0);

    return lamba.map((l) => {
      const x = ONE.div(
        ONE.add(a.mul(l))
          .add(c.mul(l))
          .add(NEG_ONE.mul(b.square()).mul(l.square()))
          .add(a.mul(c).mul(l.square())),
      ).mul(
        x0
          .add(NEG_ONE.mul(d).mul(l))
          .add(b.mul(e).mul(l.square()))
          .add(c.mul(l).mul(x0))
          .add(NEG_ONE.mul(b).mul(l).mul(y0))
          .add(NEG_ONE.mul(c).mul(d).mul(l.square())),
      );

      const y = ONE.div(
        ONE.add(a.mul(l))
          .add(c.mul(l))
          .add(NEG_ONE.mul(b.square()).mul(l.square()))
          .add(a.mul(c).mul(l.square())),
      ).mul(
        y0
          .add(NEG_ONE.mul(e).mul(l))
          .add(a.mul(l).mul(y0))
          .add(b.mul(d).mul(l.square()))
          .add(NEG_ONE.mul(a).mul(e).mul(l.square()))
          .add(NEG_ONE.mul(b).mul(l).mul(x0)),
      );
      return new Point(x, y);
    });
  }

  private sign(point: Point): Num {
    const r = new RowVec3(point.x, point.y, ONE);
    const c = r.transpose();

    const v = r.product(this._matrix).dot(c);
    return v.sign();
  }

  distanceTo(point: Point): Num {
    const distances = this.candidatePoints(point).map((candidate) =>
      candidate.vecTo(point).norm(),
    );
    return min(...(distances as [Num])).mul(this.sign(point));
  }
}

export function circleConic(radius: Num): Conic {
  return new Conic(scalingTransform(radius, radius).reverse());
}

export function ellipseConic(majorRadius: Num, minorRadius: Num): Conic {
  return new Conic(scalingTransform(majorRadius, minorRadius).reverse());
}
