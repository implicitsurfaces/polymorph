import * as m from "./matrix";

const DEFAULT_EPSILON = 1e-10;

// Inlined squared norm calculation to reduce function-call overhead.
function squaredNorm(arr) {
  let sum = 0;
  for (let i = 0, len = arr.length; i < len; i++) {
    let v = arr[i];
    sum += v * v;
  }
  return sum;
}

// Fast safe division inlined.
function safeDivide(numerator, denominator) {
  return denominator === 0
    ? numerator / (denominator + 1e-15)
    : numerator / denominator;
}

/**
 * Minimizes the objective using the BFGS algorithm.
 *
 * @param {Function} getResidualsAndGradient - A function that takes a vector x and returns
 *   an object { vals, grad } where vals is an array of residuals and grad is the gradient.
 * @param {number[]} x0 - The initial guess.
 * @param {object} [options] - Options: { epsilon, maxSteps }.
 * @returns {number[]} - The optimized vector.
 */
export function bfgs(
  getResidualsAndGradient,
  x0,
  { epsilon = DEFAULT_EPSILON, maxSteps = 1000 } = {},
) {
  // If there are no residuals, return the initial guess immediately.
  if (getResidualsAndGradient(x0).vals.length === 0) return x0;

  // Local caching of residuals and gradient functions.
  const getResiduals = (x) => getResidualsAndGradient(x).vals;
  const getGradient = (x) => getResidualsAndGradient(x).grad;

  return bfgsSolver(getResiduals, getGradient, x0, maxSteps, epsilon);
}

/**
 * Core BFGS solver with micro-optimizations.
 *
 * @param {Function} residualFunc - Function returning residuals given x.
 * @param {Function} gradientFunc - Function returning gradient given x.
 * @param {number[]} x0 - Initial guess.
 * @param {number} maxIterations - Maximum iterations allowed.
 * @param {number} tolerance - Convergence tolerance.
 * @returns {number[]} - The optimized vector.
 */
function bfgsSolver(
  residualFunc,
  gradientFunc,
  x0,
  maxIterations = 1000,
  tolerance = 1e-5,
) {
  const n = x0.length;
  let inverseHessian = m.identityMatrix(n);
  let xCurrent = x0.slice(); // shallow copy for safety
  let iteration = 0;

  while (iteration < maxIterations) {
    const currentGradient = gradientFunc(xCurrent);
    // Compute search direction: p = -inverseHessian * currentGradient.
    // Avoid an extra allocation by mapping inline.
    const p = m.multiplyMatrixVector(inverseHessian, currentGradient);
    for (let i = 0; i < p.length; i++) {
      p[i] = -p[i];
    }
    const stepSize = lineSearch(residualFunc, gradientFunc, xCurrent, p);
    const step = m.scaleVector(p, stepSize);
    const xNext = m.addVectors(xCurrent, step);

    // If the step is small enough, exit early.
    if (squaredNorm(step) < tolerance * tolerance) break;

    const nextGradient = gradientFunc(xNext);
    const gradDiff = m.subtractVectors(nextGradient, currentGradient);
    const dotSy = m.dotProduct(gradDiff, step);
    const rho = safeDivide(1, dotSy);
    // Update inverse Hessian using rank-two update:
    const outerSS = m.outerProduct(step, step);
    const term1 = m.scaleMatrix(outerSS, rho);
    const H_y = m.multiplyMatrixVector(inverseHessian, gradDiff);
    const dotYHy = m.dotProduct(gradDiff, H_y);
    const term2 = m.scaleMatrix(
      m.outerProduct(H_y, H_y),
      safeDivide(1, dotYHy),
    );
    inverseHessian = m.addMatrices(
      m.subtractMatrices(inverseHessian, term2),
      term1,
    );

    xCurrent = xNext;
    iteration++;
  }

  return xCurrent;
}

/**
 * A fast line search routine using quadratic interpolation.
 *
 * Attempts to bracket a step length that minimizes the squared norm of the residuals
 * along the given direction, then uses quadratic interpolation.
 *
 * @param {Function} residualFunc - Function returning residuals given x.
 * @param {Function} gradientFunc - Function returning gradient given x.
 * @param {number[]} x0 - Current point.
 * @param {number[]} direction - Descent direction.
 * @returns {number} - The step size.
 */
function lineSearch(residualFunc, gradientFunc, x0, direction) {
  let alpha1 = 0.0;
  let alpha2 = 1.0;
  let alpha3 = 2.0 * alpha2;
  let alphaStar;
  const maxStep = 100;

  // Cache f(x0) so we do not recompute it repeatedly.
  const f1 = squaredNorm(residualFunc(x0));

  // Compute f at alpha2.
  let xTest = m.addVectors(x0, m.scaleVector(direction, alpha2));
  let f2 = squaredNorm(residualFunc(xTest));

  // Compute f at alpha3.
  xTest = m.addVectors(x0, m.scaleVector(direction, alpha3));
  let f3 = squaredNorm(residualFunc(xTest));

  // Bracket the minimum: we expect f1 > f2 < f3.
  while (f2 > f1 || f2 > f3) {
    if (f2 > f1) {
      alpha3 = alpha2;
      f3 = f2;
      alpha2 *= 0.5;
      xTest = m.addVectors(x0, m.scaleVector(direction, alpha2));
      f2 = squaredNorm(residualFunc(xTest));
    } else if (f2 > f3) {
      if (alpha3 >= maxStep) break;
      alpha2 = alpha3;
      f2 = f3;
      alpha3 *= 2.0;
      xTest = m.addVectors(x0, m.scaleVector(direction, alpha3));
      f3 = squaredNorm(residualFunc(xTest));
    }
  }

  // Quadratic interpolation to find a better estimate.
  const numerator = (alpha2 - alpha1) * (f1 - f3);
  const denominator = 3 * (f1 - 2 * f2 + f3);
  alphaStar = alpha2 + safeDivide(numerator, denominator);
  if (alphaStar >= alpha3 || alphaStar <= alpha1) {
    alphaStar = alpha2;
  }
  if (alphaStar > maxStep) alphaStar = maxStep;
  if (isNaN(alphaStar)) alphaStar = 0.0;

  return alphaStar;
}
