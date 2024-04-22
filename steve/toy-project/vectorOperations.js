import { draw } from "pantograph2d";

export const squareDistance = ([x0, y0], [x1, y1] = [0, 0]) => {
  return (x0 - x1) ** 2 + (y0 - y1) ** 2;
};

export const distance = (p0, p1 = [0, 0]) => {
  return Math.sqrt(squareDistance(p0, p1));
};

export const angle = ([x0, y0], [x1, y1] = [0, 0]) => {
  return Math.atan2(y1 * x0 - y0 * x1, x0 * x1 + y0 * y1);
};

export function vecLength([x, y]) {
  return Math.sqrt(x * x + y * y);
}

export function angleBetween(v0, v1) {
  return Math.acos(dotProduct(v0, v1) / (vecLength(v0) * vecLength(v1)));
}

export function crossProduct([x0, y0], [x1, y1]) {
  return x0 * y1 - y0 * x1;
}

export function dotProduct([x0, y0], [x1, y1]) {
  return x0 * x1 + y0 * y1;
}

export function perpendicular(v) {
  return [-v[1], v[0]];
}

export const threePointsArc = (p0, p1, p2) => {
  const d = draw(p0).threePointsArcTo(p2, p1);
  return d.pendingSegments[0];
};

export const tgt = (p0, p1, p2) => {
  const [x, y] = threePointsArc(p0, p1, p2).tangentAt(p1);
  return Math.atan2(y, x);
};
