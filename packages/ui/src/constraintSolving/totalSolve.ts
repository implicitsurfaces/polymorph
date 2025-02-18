// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck

import { createConstraintFn } from "./compile/createConstraintFn.js";
import { levenbergMarquardt } from "./optimizers/levenbergMarquardt.js";

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
    } else if (constraint.type === "distance") {
      eqs.push(ppDist(...constraint.params, constraint.value));
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

function angle(p0, p1, p2, p3, degrees) {
  const minus = (a, b) => [`(${a[0]}-${b[0]})`, `(${a[1]}-${b[1]})`];
  const dot = (a, b) => `((${a[0]}*${b[0]}) + (${a[1]}*${b[1]}))`;
  const norm = (a) => `sqrt( ${a[0]}^2 + ${a[1]}^2 )`;

  const l1p1x = `${p0}_x`;
  const l1p1y = `${p0}_y`;
  const a = [l1p1x, l1p1y];

  const l1p2x = `${p1}_x`;
  const l1p2y = `${p1}_y`;
  const b = [l1p2x, l1p2y];

  const l2p1x = `${p2}_x`;
  const l2p1y = `${p2}_y`;
  const c = [l2p1x, l2p1y];

  const l2p2x = `${p3}_x`;
  const l2p2y = `${p3}_y`;

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

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function lpDist(p0, p1, p2, dist) {
  const lp1x = `${p0}_x`;
  const lp1y = `${p0}_y`;

  const lp2x = `${p1}_x`;
  const lp2y = `${p1}_y`;

  const px = `${p2}_x`;
  const py = `${p2}_y`;

  const top = `sqrt( ((${lp2y} - ${lp1y})*${px} - (${lp2x} - ${lp1x})*${py} + ${lp2x} * ${lp1y} - ${lp2y} * ${lp1x})^2)`;
  const bottom = `sqrt( (${lp2x} - ${lp1x})^2 + (${lp2y}-${lp1y})^2 )`;

  return `${top}/${bottom} - ${dist.toFixed(PRECISION)}`;
}
