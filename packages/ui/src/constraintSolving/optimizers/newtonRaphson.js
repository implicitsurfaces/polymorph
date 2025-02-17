import { lusolve } from "./lusolve.js";
import * as m from "./matrix.js";

const EPSILON = 1e-8;

export function newtonRaphson(
  getValDers,
  variableValues,
  { stepSize = 0.1, epsilon = EPSILON, maxSteps = 1000 } = {},
) {
  let x = variableValues;

  for (let i = 0; i < maxSteps; i++) {
    const { vals, jacobian } = getValDers(x);

    if (vals.length === 0) return x;

    const delta = solveNormalEquations(jacobian, vals);

    // const term2 = m.transposeMatrix([m.scaleVector(vals, -1)]);
    // for (let i = 0; i < jacobian.length; i++) {
    //   jacobian[i][i] += 1e-8;
    // }
    // const delta = lusolve(jacobian, term2);

    for (let j = 0; j < delta.length; j++) {
      x[j] += delta[j];
    }

    // Check if the update is within the tolerance
    if (Math.hypot(...delta) < epsilon) break;
    // if (Math.hypot(...grad) < epsilon) break;
  }

  return x;
}

function solveNormalEquations(J, f) {
  const JT = m.transposeMatrix(J);
  const H = m.multiplyMatrices(JT, J);
  const g = m.multiplyMatrixVector(JT, f);

  // Regularization: Add a small value to the diagonal of H
  for (let i = 0; i < H.length; i++) {
    H[i][i] += 1e-8;
  }

  const term2 = m.scaleVector(g, -1);

  return lusolve(H, term2);
}
