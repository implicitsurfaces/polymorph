/**
 * Marching Squares Algorithm Implementation
 *
 * This library implements the marching squares algorithm for rendering
 * implicit functions in 2D space.
 */

import { Point2D } from "./primitives";

export type Point = {
  x: number;
  y: number;
};

export type LineSegment = {
  start: Point;
  end: Point;
};

export type Grid = {
  values: number[][];
  width: number;
  height: number;
  cellSize: number;
  origin: Point;
};

export type ImplicitFunction = (x: number, y: number) => number;

/**
 * Creates a grid of values by sampling the implicit function
 */
export function createGrid(
  fn: ImplicitFunction,
  width: number,
  height: number,
  cellSize: number,
  originX: number = 0,
  originY: number = 0,
): Grid {
  const gridWidth = Math.ceil(width / cellSize) + 1;
  const gridHeight = Math.ceil(height / cellSize) + 1;

  const values: number[][] = [];

  for (let y = 0; y < gridHeight; y++) {
    const row: number[] = [];
    for (let x = 0; x < gridWidth; x++) {
      const worldX = originX + x * cellSize;
      const worldY = originY + y * cellSize;
      row.push(fn(worldX, worldY));
    }
    values.push(row);
  }

  return {
    values,
    width: gridWidth,
    height: gridHeight,
    cellSize,
    origin: { x: originX, y: originY },
  };
}

/**
 * Performs linear interpolation between two points based on the threshold value
 */
function interpolate(
  p1: Point,
  p2: Point,
  v1: number,
  v2: number,
  threshold: number,
): Point {
  if (Math.abs(v1 - v2) < 1e-10) {
    return { x: p1.x, y: p1.y };
  }

  const t = (threshold - v1) / (v2 - v1);
  return {
    x: p1.x + t * (p2.x - p1.x),
    y: p1.y + t * (p2.y - p1.y),
  };
}

/**
 * Runs the marching squares algorithm on a grid with the given threshold
 */
export function marchingSquares(grid: Grid, threshold: number): LineSegment[] {
  const segments: LineSegment[] = [];

  for (let y = 0; y < grid.height - 1; y++) {
    for (let x = 0; x < grid.width - 1; x++) {
      // Get the values at the four corners of the current cell
      const tl = grid.values[y][x];
      const tr = grid.values[y][x + 1];
      const bl = grid.values[y + 1][x];
      const br = grid.values[y + 1][x + 1];

      // Determine the case based on which corners are above the threshold
      let caseIndex = 0;
      if (tl > threshold) caseIndex |= 8;
      if (tr > threshold) caseIndex |= 4;
      if (br > threshold) caseIndex |= 2;
      if (bl > threshold) caseIndex |= 1;

      // Skip if all corners are above or below the threshold
      if (caseIndex === 0 || caseIndex === 15) continue;

      // Calculate the cell's corners in world coordinates
      const cellX = grid.origin.x + x * grid.cellSize;
      const cellY = grid.origin.y + y * grid.cellSize;

      const topLeft: Point = { x: cellX, y: cellY };
      const topRight: Point = { x: cellX + grid.cellSize, y: cellY };
      const bottomLeft: Point = { x: cellX, y: cellY + grid.cellSize };
      const bottomRight: Point = {
        x: cellX + grid.cellSize,
        y: cellY + grid.cellSize,
      };

      // Calculate the interpolated points
      const top = interpolate(topLeft, topRight, tl, tr, threshold);

      const right = interpolate(topRight, bottomRight, tr, br, threshold);

      const bottom = interpolate(bottomLeft, bottomRight, bl, br, threshold);

      const left = interpolate(topLeft, bottomLeft, tl, bl, threshold);

      // Connect the points based on the case
      switch (caseIndex) {
        case 1:
          segments.push({ start: left, end: bottom });
          break;
        case 2:
          segments.push({ start: bottom, end: right });
          break;
        case 3:
          segments.push({ start: left, end: right });
          break;
        case 4:
          segments.push({ start: top, end: right });
          break;
        case 5:
          segments.push({ start: left, end: top });
          segments.push({ start: bottom, end: right });
          break;
        case 6:
          segments.push({ start: top, end: bottom });
          break;
        case 7:
          segments.push({ start: left, end: top });
          break;
        case 8:
          segments.push({ start: left, end: top });
          break;
        case 9:
          segments.push({ start: top, end: bottom });
          break;
        case 10:
          segments.push({ start: left, end: bottom });
          segments.push({ start: top, end: right });
          break;
        case 11:
          segments.push({ start: top, end: right });
          break;
        case 12:
          segments.push({ start: left, end: right });
          break;
        case 13:
          segments.push({ start: bottom, end: right });
          break;
        case 14:
          segments.push({ start: left, end: bottom });
          break;
      }
    }
  }

  return segments;
}

/**
 * Generates contour lines for multiple thresholds
 */
export function generateContours(
  fn: ImplicitFunction,
  width: number,
  height: number,
  cellSize: number,
  thresholds: number[],
  originX: number = 0,
  originY: number = 0,
): { threshold: number; segments: LineSegment[] }[] {
  const grid = createGrid(fn, width, height, cellSize, originX, originY);

  return thresholds.map((threshold) => ({
    threshold,
    segments: marchingSquares(grid, threshold),
  }));
}

/**
 * Represents a continuous path
 */
export type Path = Point[];

/**
 * Stitches line segments into continuous paths
 */
export function stitchSegments(
  segments: LineSegment[],
  tolerance: number = 0.001,
): Path[] {
  if (segments.length === 0) return [];

  // Clone segments to avoid modifying the original array
  const remainingSegments = segments.map((segment) => ({
    start: { ...segment.start },
    end: { ...segment.end },
  }));

  const paths: Path[] = [];

  // Helper function to check if two points are close enough to be considered the same
  const arePointsEqual = (p1: Point, p2: Point): boolean => {
    return (
      Math.abs(p1.x - p2.x) < tolerance && Math.abs(p1.y - p2.y) < tolerance
    );
  };

  // Helper function to find a segment that connects to the given point
  const findConnectingSegment = (point: Point, startIndex: number): number => {
    for (let i = startIndex; i < remainingSegments.length; i++) {
      const segment = remainingSegments[i];
      if (
        arePointsEqual(point, segment.start) ||
        arePointsEqual(point, segment.end)
      ) {
        return i;
      }
    }
    return -1;
  };

  while (remainingSegments.length > 0) {
    // Start a new path with the first segment
    const firstSegment = remainingSegments[0];
    const currentPath: Point[] = [
      { ...firstSegment.start },
      { ...firstSegment.end },
    ];
    remainingSegments.splice(0, 1);

    let pathClosed = false;

    // Try to extend the path as much as possible
    while (!pathClosed && remainingSegments.length > 0) {
      const lastPoint = currentPath[currentPath.length - 1];
      const firstPoint = currentPath[0];

      // Check if the path has closed (last point connects to first point)
      if (arePointsEqual(lastPoint, firstPoint) && currentPath.length > 2) {
        pathClosed = true;
        break;
      }

      // Try to find a connecting segment
      const nextSegmentIndex = findConnectingSegment(lastPoint, 0);
      if (nextSegmentIndex === -1) break;

      const nextSegment = remainingSegments[nextSegmentIndex];
      remainingSegments.splice(nextSegmentIndex, 1);

      // Add the next point to the path (avoiding duplicates)
      if (arePointsEqual(lastPoint, nextSegment.start)) {
        currentPath.push({ ...nextSegment.end });
      } else {
        currentPath.push({ ...nextSegment.start });
      }
    }

    // Add the completed path
    paths.push(currentPath);
  }

  return paths;
}

/**
 * Simplifies a path by removing points that lie on straight lines
 */
export function simplifyPath(path: Path, tolerance: number = 0.001): Path {
  if (path.length <= 2) return [...path];

  const result: Path = [path[0]];

  for (let i = 1; i < path.length - 1; i++) {
    const prev = result[result.length - 1];
    const current = path[i];
    const next = path[i + 1];

    // Check if current point lies on a straight line between prev and next
    const dx1 = current.x - prev.x;
    const dy1 = current.y - prev.y;
    const dx2 = next.x - current.x;
    const dy2 = next.y - current.y;

    // Normalize vectors
    const len1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
    const len2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);

    // Avoid division by zero
    if (len1 < tolerance || len2 < tolerance) {
      continue;
    }

    const nx1 = dx1 / len1;
    const ny1 = dy1 / len1;
    const nx2 = dx2 / len2;
    const ny2 = dy2 / len2;

    // If directions are very close, the point lies on a straight line
    const dotProduct = nx1 * nx2 + ny1 * ny2;
    if (Math.abs(dotProduct - 1) > tolerance) {
      result.push(current);
    }
  }

  // Always add the last point
  result.push(path[path.length - 1]);

  return result;
}

/**
 * Utility function to convert line segments to SVG path data
 */
export function segmentsToSVGPath(segments: LineSegment[]): string {
  if (segments.length === 0) return "";

  return segments
    .map(
      (segment) =>
        `M${segment.start.x},${segment.start.y} L${segment.end.x},${segment.end.y}`,
    )
    .join(" ");
}

/**
 * Converts stitched paths to SVG path data
 */
export function pathsToSVGPath(paths: Path[]): string {
  if (paths.length === 0) return "";

  return paths
    .map((path) => {
      if (path.length === 0) return "";

      const d = path.reduce((acc, point, index) => {
        if (index === 0) {
          return `M${point.x},${-point.y}`;
        }
        return `${acc} L${point.x},${-point.y}`;
      }, "");

      // Close the path if the first and last points are the same
      const firstPoint = path[0];
      const lastPoint = path[path.length - 1];
      if (
        Math.abs(firstPoint.x - lastPoint.x) < 0.001 &&
        Math.abs(firstPoint.y - lastPoint.y) < 0.001
      ) {
        return `${d} Z`;
      }

      return d;
    })
    .join(" ");
}

/**
 * Handle saddle point ambiguity in the marching squares algorithm
 */
export function marchingSquaresWithAmbiguityResolution(
  grid: Grid,
  threshold: number,
  interpolationMethod: "linear" | "bilinear" = "linear",
): LineSegment[] {
  const segments: LineSegment[] = [];

  for (let y = 0; y < grid.height - 1; y++) {
    for (let x = 0; x < grid.width - 1; x++) {
      // Get the values at the four corners of the current cell
      const tl = grid.values[y][x];
      const tr = grid.values[y][x + 1];
      const bl = grid.values[y + 1][x];
      const br = grid.values[y + 1][x + 1];

      // Determine the case based on which corners are above the threshold
      let caseIndex = 0;
      if (tl > threshold) caseIndex |= 8;
      if (tr > threshold) caseIndex |= 4;
      if (br > threshold) caseIndex |= 2;
      if (bl > threshold) caseIndex |= 1;

      // Skip if all corners are above or below the threshold
      if (caseIndex === 0 || caseIndex === 15) continue;

      // Calculate the cell's corners in world coordinates
      const cellX = grid.origin.x + x * grid.cellSize;
      const cellY = grid.origin.y + y * grid.cellSize;

      const topLeft: Point = { x: cellX, y: cellY };
      const topRight: Point = { x: cellX + grid.cellSize, y: cellY };
      const bottomLeft: Point = { x: cellX, y: cellY + grid.cellSize };
      const bottomRight: Point = {
        x: cellX + grid.cellSize,
        y: cellY + grid.cellSize,
      };

      // Calculate the interpolated points
      const top = interpolate(topLeft, topRight, tl, tr, threshold);

      const right = interpolate(topRight, bottomRight, tr, br, threshold);

      const bottom = interpolate(bottomLeft, bottomRight, bl, br, threshold);

      const left = interpolate(topLeft, bottomLeft, tl, bl, threshold);

      // Handle ambiguous cases (5 and 10) based on interpolation method
      if (
        (caseIndex === 5 || caseIndex === 10) &&
        interpolationMethod === "bilinear"
      ) {
        // Calculate average value at the center
        const centerValue = (tl + tr + bl + br) / 4;

        if (caseIndex === 5) {
          if (centerValue > threshold) {
            // Connect top-left with top-right and bottom-left with bottom-right
            segments.push({ start: left, end: top });
            segments.push({ start: bottom, end: right });
          } else {
            // Connect top-left with bottom-left and top-right with bottom-right
            segments.push({ start: left, end: bottom });
            segments.push({ start: top, end: right });
          }
        } else {
          // caseIndex === 10
          if (centerValue > threshold) {
            // Connect top-left with bottom-left and top-right with bottom-right
            segments.push({ start: left, end: bottom });
            segments.push({ start: top, end: right });
          } else {
            // Connect top-left with top-right and bottom-left with bottom-right
            segments.push({ start: left, end: top });
            segments.push({ start: bottom, end: right });
          }
        }
      } else {
        // Handle all cases (including non-ambiguous ones)
        switch (caseIndex) {
          case 1:
            segments.push({ start: left, end: bottom });
            break;
          case 2:
            segments.push({ start: bottom, end: right });
            break;
          case 3:
            segments.push({ start: left, end: right });
            break;
          case 4:
            segments.push({ start: top, end: right });
            break;
          case 5:
            segments.push({ start: left, end: top });
            segments.push({ start: bottom, end: right });
            break;
          case 6:
            segments.push({ start: top, end: bottom });
            break;
          case 7:
            segments.push({ start: left, end: top });
            break;
          case 8:
            segments.push({ start: left, end: top });
            break;
          case 9:
            segments.push({ start: top, end: bottom });
            break;
          case 10:
            segments.push({ start: left, end: bottom });
            segments.push({ start: top, end: right });
            break;
          case 11:
            segments.push({ start: top, end: right });
            break;
          case 12:
            segments.push({ start: left, end: right });
            break;
          case 13:
            segments.push({ start: bottom, end: right });
            break;
          case 14:
            segments.push({ start: left, end: bottom });
            break;
        }
      }
    }
  }

  return segments;
}

/**
 * Enhanced contour generation function that produces continuous paths
 */
export function generateEnhancedContours(
  fn: ImplicitFunction,
  width: number,
  height: number,
  cellSize: number,
  thresholds: number[],
  originX: number = 0,
  originY: number = 0,
  options: {
    ambiguityResolution?: "linear" | "bilinear";
    simplifyPaths?: boolean;
  } = {},
): { threshold: number; paths: Path[] }[] {
  const grid = createGrid(fn, width, height, cellSize, originX, originY);

  return thresholds.map((threshold) => {
    // Use enhanced marching squares if bilinear interpolation is requested
    const segments =
      options.ambiguityResolution === "bilinear"
        ? marchingSquaresWithAmbiguityResolution(grid, threshold, "bilinear")
        : marchingSquares(grid, threshold);

    // Stitch segments into continuous paths
    let paths = stitchSegments(segments);

    // Optionally simplify paths
    if (options.simplifyPaths) {
      paths = paths.map((path) => simplifyPath(path));
    }

    return {
      threshold,
      paths,
    };
  });
}

/**
 * Example usage:
 *
 * // Define an implicit function (a circle)
 * const circle = (x: number, y: number): number => {
 *   return Math.sqrt((x - 150) ** 2 + (y - 150) ** 2) - 100;
 * };
 *
 * // Generate enhanced contours with continuous paths
 * const contours = generateEnhancedContours(
 *   circle, 300, 300, 5, [0, 20, 40],
 *   0, 0,
 *   { ambiguityResolution: 'bilinear', simplifyPaths: true }
 * );
 *
 * // Convert to SVG paths
 * const svgPaths = contours.map(contour => ({
 *   threshold: contour.threshold,
 *   svgPath: pathsToSVGPath(contour.paths)
 * }));
 *
 * // Render as an SVG
 * const svg = `
 * <svg width="300" height="300">
 *   ${svgPaths.map((item, i) =>
 *     `<path d="${item.svgPath}" fill="none" stroke="hsl(${i * 120}, 100%, 50%)" stroke-width="1" />`
 *   ).join('\n')}
 * </svg>`;
 */

type ShapeLike = { levelSet(point: Point2D): number };
export type SVGRenderOptions = {
  width?: number;
  height?: number;
  originX?: number;
  originY?: number;
  cellSize?: number | null;
  cellFactor?: number;
  threshold?: number;
};
type ShapeConfig = {
  shape: ShapeLike;
  color?: string;
  strokeWidth?: number;
};

export type ShapeInput = ShapeConfig | ShapeLike;

function viewBox({
  width = 2,
  height = 2,
  originX = -1,
  originY = -1,
}: SVGRenderOptions = {}) {
  return `${originX} ${originY} ${width} ${height}`;
}

function shapeToSVGPath(
  shape: ShapeInput,
  {
    width = 2,
    height = 2,
    originX = -1,
    originY = -1,
    cellSize = null,
    cellFactor = 1,
    threshold = 0,
  }: SVGRenderOptions = {},
) {
  const size = Math.min(width, height);

  const config: ShapeConfig = "shape" in shape ? shape : { shape };

  const contours = generateEnhancedContours(
    (x, y) => config.shape.levelSet(new Point2D(x, y)),
    width,
    height,
    cellSize ? cellSize : (size / 200) * cellFactor,
    [threshold],
    originX,
    originY,
  );

  const paths = contours.map((c) => pathsToSVGPath(c.paths));
  return paths
    .map(
      (svgPath) =>
        `<path d="${svgPath}" fill="none" stroke="${config.color || "grey"}" stroke-width="${(size / 400) * (config.strokeWidth || 1)}"/>`,
    )
    .join("\n");
}

export function renderAsSVG(
  shape: ShapeInput | ShapeInput[],
  options: SVGRenderOptions = {},
) {
  const size = Math.min(options.width || 2, options.height || 2);

  if (!Array.isArray(shape)) {
    shape = [shape];
  }

  return `<svg width="750px" height="750px" viewBox="${viewBox(options)}" xmlns="http://www.w3.org/2000/svg">
    <circle cx="0" cy="0" r="${size / 200}" fill="black" />
  ${shape
    .map((s) => {
      if (s instanceof Point2D) {
        return `<circle cx="${s.x}" cy="${-s.y}" r="${size / 200}" fill="none" stroke="black" stroke-width="${size / 400}" />`;
      }
      return shapeToSVGPath(s, options);
    })
    .join("\n")}
</svg>`;
}
