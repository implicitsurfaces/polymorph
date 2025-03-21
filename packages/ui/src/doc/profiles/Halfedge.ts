import { EdgeNode } from "../EdgeNode";
import { Document } from "../Document";
import { Point } from "../Point";

export type HalfedgeId = string;

/**
 * Any type that can be passed to the constructor of `Halfedge`.
 *
 * If no `direction` is provided, it is assumed to be `true`, that is,
 * an edge is by default considered in its intrinsic direction, from its
 * start point to its end point.
 */
export type HalfedgeLike =
  | EdgeNode
  | [EdgeNode, boolean]
  | { edge: EdgeNode; direction: boolean };

/**
 * A halfedge is an edge together with a given boolean `direction`:
 *
 * - If the direction is `true`, then the edge is considered in its intrinsic
 *   parameterized direction, from its start point to its end point.
 *
 * - If the direction is `false`, then the edge is considered in its opposite
 *   parameterized direction, from its end point to its start point.
 */
export class Halfedge {
  readonly edge: EdgeNode;
  readonly direction: boolean;

  /**
   * Constructs a halfedge.
   *
   * If provided, the optional second argument (`direction`) overrides any
   * direction already provided by the first argument (`other`).
   *
   * If no direction is provided (neither from the second argument nor
   * `other`), then the direction is assumed to be `true`.
   *
   * Examples:
   *
   * ```
   * new Halfedge(edge);                // => [edge, true]
   * new Halfedge(edge, true);          // => [edge, true]
   * new Halfedge(edge, false);         // => [edge, false]
   *
   * new Halfedge(halfedge);            // => [halfedge.edge, halfedge.direction]
   * new Halfedge(halfedge, true);      // => [halfedge.edge, true]
   * new Halfedge(halfedge, false);     // => [halfedge.edge, false]
   *
   * new Halfedge([edge, direction]);        // => [edge, direction]
   * new Halfedge([edge, direction], false); // => [edge, false]
   * new Halfedge([edge, direction], true);  // => [edge, true]
   *
   * new Halfedge({edge: e, direction: d});        // => [e, d]
   * new Halfedge({edge: e, direction: d}, false); // => [e, false]
   * new Halfedge({edge: e, direction: d}, true);  // => [e, true]
   * ```
   */
  constructor(other: HalfedgeLike, direction?: boolean) {
    if (other instanceof EdgeNode) {
      this.edge = other;
      this.direction = direction ?? true;
    } else if (Array.isArray(other)) {
      this.edge = other[0];
      this.direction = direction ?? other[1];
    } else {
      this.edge = other.edge;
      this.direction = direction ?? other.direction ?? true;
    }
  }

  static fromId(id: HalfedgeId, doc: Document): Halfedge | undefined {
    let direction = true;
    let edgeId = id;
    const lastChar = id.slice(-1);
    if (lastChar === "*") {
      direction = false;
      edgeId = id.slice(0, -1);
    }
    const edge = doc.getNode(edgeId, EdgeNode);
    return edge ? new Halfedge(edge, direction) : undefined;
  }

  id(): HalfedgeId {
    return this.direction ? this.edge.id : `${this.edge.id}*`;
  }

  startPoint(): Point {
    return this.direction ? this.edge.startPoint : this.edge.endPoint;
  }

  endPoint(): Point {
    return this.direction ? this.edge.endPoint : this.edge.startPoint;
  }

  opposite(): Halfedge {
    return new Halfedge(this.edge, !this.direction);
  }
}
