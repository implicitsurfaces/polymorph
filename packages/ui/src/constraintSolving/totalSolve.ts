// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck

import { createConstraintFn } from "./compile/createConstraintFn";
import { levenbergMarquardt } from "./optimizers/levenbergMarquardt";

export function totalSolve(constraints, currentParams) {
  // console.log({ constraints, currentParams });

  const eqs = [];

  constraints.forEach((constraint) => {
    if (constraint.type === "angle") {
      eqs.push(angle(...constraint.params, constraint.value));
    } else if (constraint.type === "perpendicular") {
      eqs.push(angle(...constraint.params, 90));
    } else if (constraint.type === "parallel") {
      eqs.push(angle(...constraint.params, 0));
    } else if (constraint.type === "pointToPointDistance") {
      eqs.push(ppDist(...constraint.params, constraint.value));
    } else if (constraint.type === "lineToPointDistance") {
      eqs.push(lpDist(...constraint.params, constraint.value));
    } else {
      console.log("missing", constraint);
    }
  });

  const variableNames = [];
  const initialValues = [];

  for (const id in currentParams) {
    const val = currentParams[id];
    variableNames.push(id);
    initialValues.push(val);
  }

  const fn = createConstraintFn(eqs, variableNames);

  const solution = levenbergMarquardt(fn, initialValues);

  const finalValues = {};

  for (let i = 0; i < variableNames.length; i += 1) {
    const id = variableNames[i];

    finalValues[id] = solution[i];
  }

  // console.log({ eqs, variableNames, initialValues, solution });

  return finalValues;
}

const PRECISION = 2;

function ppDist(p0x, p0y, p1x, p1y, dist) {
  return `${(dist ** 2).toFixed(
    PRECISION,
  )} - ((${p1x}-${p0x})^2+(${p1y}-${p0y})^2)`;
}

function angle(
  l1p1x,
  l1p1y,
  l1p2x,
  l1p2y,
  l2p1x,
  l2p1y,
  l2p2x,
  l2p2y,
  degrees,
) {
  const minus = (a, b) => [`(${a[0]}-${b[0]})`, `(${a[1]}-${b[1]})`];
  const dot = (a, b) => `((${a[0]}*${b[0]}) + (${a[1]}*${b[1]}))`;
  const norm = (a) => `sqrt( ${a[0]}^2 + ${a[1]}^2 )`;

  const a = [l1p1x, l1p1y];
  const b = [l1p2x, l1p2y];
  const c = [l2p1x, l2p1y];

  const angleRads = (degrees / 180) * Math.PI + Math.PI / 2;
  const cosTheta = `${Math.cos(angleRads).toFixed(PRECISION)}`;
  const sinTheta = `${Math.sin(angleRads).toFixed(PRECISION)}`;
  const rotatedL2p2x = `((${l2p2x} - ${l2p1x}) * ${cosTheta} - (${l2p2y} - ${l2p1y}) * ${sinTheta} + ${l2p1x})`;
  const rotatedL2p2y = `((${l2p2x} - ${l2p1x}) * ${sinTheta} + (${l2p2y} - ${l2p1y}) * ${cosTheta} + ${l2p1y})`;
  const dRot = [rotatedL2p2x, rotatedL2p2y];

  const numerator = `${dot(minus(a, b), minus(c, dRot))}`;

  // used to prevent line from collapsing
  const denominator = `( ${norm(minus(a, b))} * ${norm(minus(c, dRot))} )`;

  const final = `${numerator}/${denominator}`;

  return final;
}

function lpDist(lp1x, lp1y, lp2x, lp2y, px, py, dist) {
  const top = `sqrt( ((${lp2y} - ${lp1y})*${px} - (${lp2x} - ${lp1x})*${py} + ${lp2x} * ${lp1y} - ${lp2y} * ${lp1x})^2)`;
  const bottom = `sqrt( (${lp2x} - ${lp1x})^2 + (${lp2y}-${lp1y})^2 )`;

  return `${top}/${bottom} - ${dist.toFixed(PRECISION)}`;
}
