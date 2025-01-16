import { Point } from "../geom";
import { asNum, NEG_ONE, Num, ONE, ZERO } from "../num";
import { hypot, ifTruthyElse } from "../num-ops";

function pow3(x: Num) {
  return x.mul(x).mul(x);
}

function mSign(x: Num) {
  return ifTruthyElse(x.lessThan(ZERO), NEG_ONE, ONE);
}

const T0 = asNum(0.707);

export function closestPointOnEllipse(
  majorRadius: Num,
  minorRadius: Num,
  point: Point,
) {
  const px = point.x.abs();
  const py = point.y.abs();

  let tx = T0;
  let ty = T0;

  const majorRadiusSq = majorRadius.square();
  const minorRadiusSq = minorRadius.square();

  const squareDiff = majorRadiusSq.sub(minorRadiusSq);
  const negSquareDiff = squareDiff.neg();

  for (let i = 0; i < 3; i++) {
    const x = majorRadius.mul(tx);
    const y = minorRadius.mul(ty);

    const ex = squareDiff.mul(pow3(tx)).div(majorRadius);
    const ey = negSquareDiff.mul(pow3(ty)).div(minorRadius);

    const rx = x.sub(ex);
    const ry = y.sub(ey);

    const qx = px.sub(ex);
    const qy = py.sub(ey);

    const r = hypot(rx, ry);
    const q = hypot(qx, qy);

    tx = qx.mul(r).div(q).add(ex).div(majorRadius).max(0).min(1);
    ty = qy.mul(r).div(q).add(ey).div(minorRadius).max(0).min(1);

    const t = hypot(tx, ty);
    tx = tx.div(t);
    ty = ty.div(t);
  }

  return new Point(
    majorRadius.mul(tx).mul(mSign(point.x)),
    minorRadius.mul(ty).mul(mSign(point.y)),
  );
}
