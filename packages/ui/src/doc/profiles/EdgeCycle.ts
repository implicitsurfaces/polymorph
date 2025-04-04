import { Halfedge, HalfedgeLike } from "./Halfedge";

export type EdgeCycleLike = HalfedgeLike[] | EdgeCycle;

/**
 * An edge cycle stores a sequence of consecutive halfedges that
 * starts and ends at the same point.
 *
 * The following invariants are enforced, where `n = cycle.halfedges.length`:
 * - `n > 0`
 * - For all i in [0..n-1]:
 *   `cycle.halfedges[i % n].endPoint() == cycle.halfedges[(i+1) % n].startPoint()`
 */
export class EdgeCycle {
  readonly halfedges: readonly Halfedge[];

  constructor(other: EdgeCycleLike) {
    if (other instanceof EdgeCycle) {
      this.halfedges = other.halfedges;
    } else {
      // Convert from from HalfedgeLike to Halfedge
      this.halfedges = other.map((h) => new Halfedge(h));

      // Check preconditions
      if (this.halfedges.length === 0) {
        throw Error(`An edge cycle must contain at least one edge.`);
      }
      let prevEndPoint = this.halfedges[this.halfedges.length - 1].endPoint();
      for (const halfedge of this.halfedges) {
        if (halfedge.startPoint() !== prevEndPoint) {
          throw Error(
            `Consecutive halfedges in an edge cycle must share their endpoint.`,
          );
        }
        prevEndPoint = halfedge.endPoint();
      }
    }
  }
}
