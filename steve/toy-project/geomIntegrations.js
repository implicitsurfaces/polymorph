import { integrate } from "./gaussLegendreIntegration.js";
import { vecLength, perpendicular, dotProduct } from "./vectorOperations.js";

export function integrateSurfaceWithRaster(shape, gridSize = 100) {
  const bbox = shape.boundingBox;
  const grid = range(gridSize).flatMap((i) =>
    range(gridSize).map((j) => [
      bbox.xMin + (i * bbox.width) / gridSize,
      bbox.yMin + (j * bbox.height) / gridSize,
    ]),
  );

  return (
    (grid
      .map((point) => (shape.contains(point) ? 1 : 0))
      .reduce((a, b) => a + b, 0) *
      bbox.width *
      bbox.height) /
    (gridSize * gridSize)
  );
}

export function integrateSurface(shape) {
  return Math.abs(
    shape.segments
      .map((segment) =>
        integrate(
          (t) =>
            dotProduct(
              segment.gradientAt(t),
              perpendicular(segment.paramPoint(t)),
            ),
          0,
          1,
        ),
      )
      .reduce((a, b) => a + b, 0) / 2,
  );
}

export function integrateLength(loop) {
  return loop.segments
    .map((segment) => integrate((t) => vecLength(segment.gradientAt(t)), 0, 1))
    .reduce((a, b) => a + b, 0);
}
