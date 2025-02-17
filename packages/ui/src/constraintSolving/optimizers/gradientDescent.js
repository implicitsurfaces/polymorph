export function gradientDescent(
  getValDers,
  initValues,
  { stepSize = 0.1, maxSteps = 500, epsilon = 0.0001 } = {},
) {
  const optimizedVals = [...initValues];

  for (let i = 0; i < maxSteps; i++) {
    const { vals, grad } = getValDers(optimizedVals);

    const error = Math.hypot(...vals);
    if (error < epsilon) break;

    grad.forEach((g, index) => {
      optimizedVals[index] -= g * stepSize;
    });
  }

  return optimizedVals;
}
