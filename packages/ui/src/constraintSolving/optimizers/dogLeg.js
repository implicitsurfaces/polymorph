import { lusolve } from "./lusolve.js";
import { choleskySolve } from "./choleskySolve.js";

import * as m from "./matrix.js";

export function dogLeg(
  getValDers,
  x0,
  { delta0 = null, epsilon, maxSteps } = {},
) {
  let k = 0;
  const kMax = maxSteps || 1000;
  const eps = epsilon || 1e-8;
  let x = x0;
  let valDers = getValDers(x);
  let f_x = valDers.vals;
  if (f_x.length === 0) return x0;

  let J_x = valDers.jacobian;
  let g = m.multiplyMatrixVector(m.transposeMatrix(J_x), f_x);
  let delta = delta0 || normInf(g) * 0.5;
  let found = normInf(f_x) <= eps || normInf(g) <= eps;

  while (!found && k < kMax) {
    k++;

    const h_gn = solveNormalEquations(J_x, f_x);
    // Cache J_x * g so we compute it only once.
    const Jg = m.multiplyMatrixVector(J_x, g);
    const normJg = m.norm2(Jg);
    const alpha = safeDivide(m.norm2(g), normJg);
    const h_sd = m.scaleVector(g, -alpha);
    const h_dl = computeHdl(h_sd, h_gn, delta);

    if (m.norm(h_dl) <= eps * (m.norm(x) + eps)) {
      found = true;
    } else {
      const x_new = m.addVectors(x, h_dl);
      // Compute getValDers(x_new) only once.
      const newValDers = getValDers(x_new);
      const f_x_new = newValDers.vals;
      const F_x = 0.5 * m.norm2(f_x);
      const F_x_new = 0.5 * m.norm2(f_x_new);
      const L_hdl = computeL(f_x, J_x, h_dl);
      const rho = safeDivide(F_x - F_x_new, F_x - L_hdl);

      if (rho > 0) {
        x = x_new;
        valDers = newValDers;
        f_x = valDers.vals;
        J_x = valDers.jacobian;
        g = m.multiplyMatrixVector(m.transposeMatrix(J_x), f_x);
        found = normInf(f_x) <= eps || normInf(g) <= eps;
      }

      if (rho > 0.75) {
        delta = Math.max(delta, 3 * m.norm(h_dl));
      } else if (rho < 0.25) {
        delta /= 2;
        if (delta <= eps * (m.norm(x) + eps)) {
          found = true;
        }
      }
    }
  }

  return x;
}

function solveNormalEquations(J, f) {
  const JT = m.transposeMatrix(J);
  const H = m.multiplyMatrices(JT, J);
  const g = m.multiplyMatrixVector(JT, f);
  // Regularize the diagonal.
  for (let i = 0; i < H.length; i++) {
    H[i][i] += 1e-8;
  }
  const rhs = m.transposeMatrix([m.scaleVector(g, -1)]);
  // Consider replacing lusolve with a Cholesky solver if H is symmetric positive definite.
  return choleskySolve(H, rhs);
}

function computeHdl(h_sd, h_gn, delta) {
  const norm_h_gn = m.norm(h_gn);
  if (norm_h_gn <= delta) return h_gn;
  const norm_h_sd = m.norm(h_sd);
  if (norm_h_sd >= delta) return m.scaleVector(h_sd, delta / norm_h_sd);

  const d = m.subtractVectors(h_gn, h_sd);
  const dNorm2 = m.norm2(d);
  const h_sd_norm2 = m.norm2(h_sd);
  const h_sd_dot_d = m.dotProduct(h_sd, d);
  const delta2 = delta * delta;
  // Solve for beta in: ||h_sd + beta*d|| = delta.
  const beta =
    (-h_sd_dot_d +
      Math.sqrt(h_sd_dot_d * h_sd_dot_d + dNorm2 * (delta2 - h_sd_norm2))) /
    dNorm2;
  return m.addVectors(h_sd, m.scaleVector(d, beta));
}

function computeL(f, J, h) {
  return 0.5 * m.norm2(m.addVectors(f, m.multiplyMatrixVector(J, h)));
}

function normInf(vector) {
  return Math.max(...vector.map(Math.abs));
}

function safeDivide(num, denom) {
  if (denom === 0) {
    console.log("dividing by zero");
    denom += 1e-15;
  }
  return num / denom;
}
