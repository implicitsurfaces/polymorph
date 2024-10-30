// Geometric types used for the UI
//
// TODO:
// - Should the UI simply use the geom types from the `sketch` package?
//   That seems overkill for typical use cases where we do not need to store an expression tree.
// - Or have another shared `geometry` package with basic geom types without an expression tree?
//
// While settling on these decision, we define here basic types without too
// much engineering.

/**
 * Represents a 2D point.
 */
export class Point {
  constructor(
    public x: number = 0,
    public y: number = 0
  ) {}
}
