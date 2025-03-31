const PRECISION = 4;

export function ppDist(p0x, p0y, p1x, p1y, dist) {
  return `${dist.toFixed(
    PRECISION,
  )} - sqrt((${p1x}-${p0x})^2+(${p1y}-${p0y})^2)`;
}

export function angle(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, degrees) {
  const minus = (a, b) => [`(${a[0]}-${b[0]})`, `(${a[1]}-${b[1]})`];
  const dot = (a, b) => `((${a[0]}*${b[0]}) + (${a[1]}*${b[1]}))`;
  const norm = (a) => `sqrt( ${a[0]}^2 + ${a[1]}^2 )`;
  const div = (a, b) => `(${a})/(${b})`;

  const a = [p0x, p0y];
  const b = [p1x, p1y];
  const c = [p2x, p2y];
  const d = [p3x, p3y];

  const angleRads = (degrees / 180) * Math.PI + Math.PI / 2;
  const cosTheta = `${Math.cos(angleRads).toFixed(PRECISION)}`;
  const sinTheta = `${Math.sin(angleRads).toFixed(PRECISION)}`;
  const rotatedL2p2x = `((${p3x} - ${p2x}) * ${cosTheta} - (${p3y} - ${p2y}) * ${sinTheta} + ${p2x})`;
  const rotatedL2p2y = `((${p3x} - ${p2x}) * ${sinTheta} + (${p3y} - ${p2y}) * ${cosTheta} + ${p2y})`;
  const dRot = [rotatedL2p2x, rotatedL2p2y];

  const numerator = `${dot(minus(a, b), minus(c, dRot))}`;

  // used to prevent line from collapsing
  const denominator = `( ${norm(minus(a, b))} * ${norm(minus(c, dRot))} )`;

  const final = `${numerator}/${denominator}`;

  return final;
}

export function lpDist(lp1x, lp1y, lp2x, lp2y, px, py, dist) {
  const abs = (x) => `sqrt( (${x})^2 )`;

  let top = `(${lp2y} - ${lp1y})*${px} - (${lp2x} - ${lp1x})*${py} + ${lp2x} * ${lp1y} - ${lp2y} * ${lp1x}`;
  let bottom = `sqrt( (${lp2x} - ${lp1x})^2 + (${lp2y} - ${lp1y})^2 )`;

  return `${abs(top)}/${bottom} - ${dist.toFixed(PRECISION)}`;
}

export function pointOnLine(lp1x, lp1y, lp2x, lp2y, px, py) {
  let determinant = `(${px} - ${lp1x})*(${lp2y} - ${lp1y}) - (${py} - ${lp1y})*(${lp2x} - ${lp1x})`;

  return `${determinant}`;
}

export function equal(l1p1x, l1p1y, l1p2x, l1p2y, l2p1x, l2p1y, l2p2x, l2p2y) {
  let d1 = `sqrt((${l1p2x}-${l1p1x})^2+(${l1p2y}-${l1p1y})^2)`;
  let d2 = `sqrt((${l2p2x}-${l2p1x})^2+(${l2p2y}-${l2p1y})^2)`;

  return `${d1} - ${d2}`;
}

export function vertical(p0x, p1x) {
  return `${p0x} - ${p1x}`;
}

export function horizontal(p0y, p1y) {
  return `${p0y} - ${p1y}`;
}

export function midpoint(lp1x, lp1y, lp2x, lp2y, px, py) {
  let midpointX = `(${lp1x} + ${lp2x}) / 2`;
  let midpointY = `(${lp1y} + ${lp2y}) / 2`;

  let constraintX = `${px} - (${midpointX})`;
  let constraintY = `${py} - (${midpointY})`;

  return `(${constraintX})^2 + (${constraintY})^2`;
}
