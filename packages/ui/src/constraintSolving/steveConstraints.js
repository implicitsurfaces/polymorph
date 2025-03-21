import { asNum, simpleEval, NumX, variable } from "sketch";

const PRECISION = 2;

export function ppDist(p0x, p0y, p1x, p1y, dist) {
  // (dist^2) - [ (p1x - p0x)^2 + (p1y - p0y)^2 ]
  const dy = variable(p1y).sub(variable(p0y));
  const dx = variable(p1x).sub(variable(p0x));
  return asNum(dist)
    .mul(dist)
    .sub(dx.mul(dx).add(dy.mul(dy)));
}

export function angle(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, degrees) {
  p0x = variable(p0x);
  p0y = variable(p0y);
  p1x = variable(p1x);
  p1y = variable(p1y);
  p2x = variable(p2x);
  p2y = variable(p2y);
  p3x = variable(p3x);
  p3y = variable(p3y);

  // Rotation angle: (degrees/180)*PI + PI/2
  const angleRads = (degrees / 180) * Math.PI + Math.PI / 2;
  const cosTheta = asNum(Math.cos(angleRads));
  const sinTheta = asNum(Math.sin(angleRads));

  // First line vector: v1 = p0 - p1
  const v1x = p0x.sub(p1x);
  const v1y = p0y.sub(p1y);

  // Second line: define p2 and p3, then rotate p3 around p2
  const dx = p3x.sub(p2x);
  const dy = p3y.sub(p2y);
  const rotatedX = dx.mul(cosTheta).sub(dy.mul(sinTheta)).add(p2x);
  const rotatedY = dx.mul(sinTheta).add(dy.mul(cosTheta)).add(p2y);
  // Second vector: v2 = p2 - rotated(p3)
  const v2x = p2x.sub(rotatedX);
  const v2y = p2y.sub(rotatedY);

  // Dot product and norms
  const numerator = v1x.mul(v2x).add(v1y.mul(v2y));
  const norm1 = v1x.mul(v1x).add(v1y.mul(v1y)).sqrt();
  const norm2 = v2x.mul(v2x).add(v2y.mul(v2y)).sqrt();

  return numerator.div(norm1.mul(norm2));
}

export function lpDist(lp1x, lp1y, lp2x, lp2y, px, py, dist) {
  lp1x = variable(lp1x);
  lp1y = variable(lp1y);
  lp2x = variable(lp2x);
  lp2y = variable(lp2y);
  px = variable(px);
  py = variable(py);

  // Compute: ((lp2y-lp1y)*px - (lp2x-lp1x)*py + lp2x*lp1y - lp2y*lp1x)^2
  const term1 = lp2y.sub(lp1y).mul(px);
  const term2 = lp2x.sub(lp1x).mul(py);
  const term3 = lp2x.mul(lp1y);
  const term4 = lp2y.mul(lp1x);
  const inner = term1.sub(term2).add(term3).sub(term4);
  const numerator = inner.mul(inner).sqrt();
  const term5 = lp2y.sub(lp1y);
  const term6 = lp2x.sub(lp1x);

  const denominator = term6.mul(term6).add(term5.mul(term5)).sqrt();

  return numerator.div(denominator).sub(asNum(dist));
}
