import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type NodeId = string;

export interface NodeOptions {
  name?: string;
}

export abstract class Node {
  readonly id: NodeId;
  name: string;

  constructor(id: NodeId, options: NodeOptions) {
    this.id = id;
    this.name = options.name !== undefined ? options.name : "Node";
  }

  abstract clone(): Node;
}

/**
 * Represents any constructible Node type.
 *
 * This is used for typing factory functions like `Document.createNode()`.
 */
type NodeType<T, Options> = (new (id: NodeId, options: Options) => T) & {
  defaultName: string;
};

/**
 * Represents any Node type, not necessarily constructible.
 *
 * This is used for functions with runtime type checks like `Document.getNode()`.
 */
type AbstractNodeType<T> = abstract new (id: NodeId, options: never) => T;

///////////////////////////////////////////////////////////////////////////////
//                               SkeletonNode

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface SkeletonNodeOptions extends NodeOptions {}

export abstract class SkeletonNode extends Node {
  constructor(id: NodeId, options: SkeletonNodeOptions) {
    super(id, options);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends SkeletonNodeOptions {
  position?: Vector2;
}

export class Point extends SkeletonNode {
  static readonly defaultName = "Point";
  position: Vector2;

  constructor(id: NodeId, options: PointOptions) {
    super(id, options);
    this.position = options.position
      ? options.position.clone()
      : new Vector2(0, 0);
  }

  clone() {
    return new Point(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               EdgeNode

export interface EdgeNodeOptions extends SkeletonNodeOptions {
  startPoint: NodeId;
  endPoint: NodeId;
}

export abstract class EdgeNode extends SkeletonNode {
  startPoint: NodeId;
  endPoint: NodeId;

  constructor(id: NodeId, options: EdgeNodeOptions) {
    super(id, options);
    this.startPoint = options.startPoint;
    this.endPoint = options.endPoint;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               LineSegment

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeNodeOptions {}

export class LineSegment extends EdgeNode {
  static readonly defaultName = "Line Segment";

  constructor(
    readonly id: NodeId,
    options: LineSegmentOptions,
  ) {
    super(id, options);
  }

  clone() {
    return new LineSegment(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               ArcFromStartTangent

export interface ArcFromStartTangentOptions extends EdgeNodeOptions {
  controlPoint?: Vector2;
}

export class ArcFromStartTangent extends EdgeNode {
  static readonly defaultName = "Arc";
  controlPoint: Vector2;

  constructor(
    readonly id: NodeId,
    options: ArcFromStartTangentOptions,
  ) {
    super(id, options);
    this.controlPoint = options.controlPoint
      ? options.controlPoint.clone()
      : new Vector2(0, 0);
  }

  clone() {
    return new ArcFromStartTangent(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  CCurve

export interface CCurveOptions extends EdgeNodeOptions {
  controlPoint?: Vector2;
}

export class CCurve extends EdgeNode {
  static readonly defaultName = "C-Curve";
  controlPoint: Vector2;

  constructor(
    readonly id: NodeId,
    options: CCurveOptions,
  ) {
    super(id, options);
    this.controlPoint = options.controlPoint
      ? options.controlPoint.clone()
      : new Vector2(0, 0);
  }

  clone() {
    return new CCurve(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  SCurve

export interface SCurveOptions extends EdgeNodeOptions {
  startControlPoint?: Vector2;
  endControlPoint?: Vector2;
}

export class SCurve extends EdgeNode {
  static readonly defaultName = "S-Curve";
  startControlPoint: Vector2;
  endControlPoint: Vector2;

  constructor(
    readonly id: NodeId,
    options: SCurveOptions,
  ) {
    super(id, options);
    this.startControlPoint = options.startControlPoint
      ? options.startControlPoint.clone()
      : new Vector2(0, 0);
    this.endControlPoint = options.endControlPoint
      ? options.endControlPoint.clone()
      : new Vector2(0, 0);
  }

  clone() {
    return new SCurve(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends NodeOptions {
  nodes?: Array<NodeId>;
}

export class Layer extends Node {
  static readonly defaultName = "Layer";
  nodes: Array<NodeId>;

  constructor(id: NodeId, options: LayerOptions) {
    super(id, options);
    this.nodes = options.nodes ? [...options.nodes] : [];
  }

  clone() {
    return new Layer(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneNodeMap(source: Map<NodeId, Node>) {
  const dest = new Map<NodeId, Node>();
  source.forEach((node, id) => {
    dest.set(id, node.clone());
  });
  return dest;
}

function sortAndRemoveDuplicates(array: Array<number>) {
  const copy = [...array];
  if (copy.length === 0) {
    return copy;
  }
  copy.sort((a, b) => a - b); // sort by increasing value (default = alphabetical)
  const res = [copy[0]];
  for (let i = 1; i < copy.length; i++) {
    if (copy[i - 1] !== copy[i]) {
      res.push(copy[i]);
    }
  }
  return res;
}

///////////////////////////////////////////////////////////////////////////////
//                               Document

/**
 * Stores all objects in the document.
 */
export class Document {
  private _nodes: Map<NodeId, Node>;

  public layers: Array<NodeId>;

  constructor(other?: Document) {
    if (other) {
      this._nodes = cloneNodeMap(other._nodes);
      this.layers = [...other.layers];
    } else {
      this._nodes = new Map();
      this.layers = [];
    }
  }

  /**
   * Returns a new document with the same content as this one.
   */
  clone(): Document {
    return new Document(this);
  }

  /**
   * Returns the node that has the given `id`, if any:
   *
   * ```
   * const node = doc.getNode(id);
   * ```
   *
   * If `type` is given as argument to this function, then this function
   * checks at runtime via `instanceof` that the node is of the given
   * `type`, and otherwise returns `undefined`:
   *
   * ```
   * const point = doc.createNode(Point);
   * const segment = doc.createNode(LineSegment);
   * const p1 = doc.getNode(point.id, Point);   // === point
   * const p2 = doc.getNode(segment.id, Point); // === undefined
   * ```
   *
   * If no `type` is given as argument to this function, but an explicit type
   * argument `T` is provided, then this function assumes the node is
   * indeed of type `T` and performs type narrowing from `Node |
   * undefined` to `T | undefined` without runtime checks. This is unsafe and
   * therefore not recommended.
   *
   * ```
   * const point = doc.createNode(Point);
   * const segment = doc.createNode(LineSegment);
   * const p1 = doc.getNode<Point>(point.id, Point);   // === point
   * const p2 = doc.getNode<Point>(segment.id, Point); // === segment as Point (bug!)
   * ```
   */
  getNode<T extends Node>(
    id: NodeId | undefined,
    type?: AbstractNodeType<T> | undefined,
  ): T | undefined {
    if (!id) {
      return undefined;
    }
    const node: Node | undefined = this._nodes.get(id);
    if (type === undefined) {
      // unchecked type narrowing
      return node as T | undefined;
    } else if (node instanceof type) {
      // checked type narrowing
      return node;
    } else {
      return undefined;
    }
  }

  /**
   * Returns an array of nodes corresponding to the given array of `ids`.
   */
  getNodes<T extends Node>(
    ids: Array<NodeId>,
    type?: AbstractNodeType<T>,
  ): Array<T> {
    const res: Array<T> = [];
    for (const id of ids) {
      const node = this.getNode(id, type);
      if (node) {
        res.push(node);
      }
    }
    return res;
  }

  /**
   * Creates and returns a new node of the given `spec` with the given `options`.
   *
   * Example:
   *
   * ```
   * const p = doc.createNode(Point, {
   *   name: "New Point",
   *   position: new Vector2(42, 12),
   * });
   * ```
   *
   * Note: this does not add the node to any layer. You typically want to:
   * 1. Mutate the `nodes` attribute of some layer after calling this function, or
   * 2. Use `createNodeInLayer()` instead of this function.
   */
  createNode<T extends Node, Options>(
    type: NodeType<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    const node = new type(id, options);
    this._nodes.set(id, node);
    return node;
  }

  foo() {
    this.createNode(Point, { name: "My Point" });
  }
  /**
   * Creates a new node of the given `spec` with the given `options`, adds it
   * as last node of the given layer, then returns it.
   *
   * If no name is provided in the options, then this function will
   * automatically a unique name withing the `layer` suitable for the given
   * `spec`, e.g., "Point 42".
   */
  createNodeInLayer<T extends Node, Options extends NodeOptions>(
    type: NodeType<T, Options>,
    layer: Layer,
    options: Options,
  ): T {
    if (options.name === undefined) {
      options.name = this.findAvailableName(
        type.defaultName + " ",
        layer.nodes,
      );
    }
    const node = this.createNode(type, options);
    layer.nodes.push(node.id);
    return node;
  }

  /**
   * Removes the node that has the given `id` from this document.
   *
   * Note: this does not remove the node from any layer. You typically want to:
   * 1. Mutate the `nodes` attribute of some layer before calling this function, or
   * 2. Use `removeNodeInLayer()` instead of this function.
   */
  removeNode(id: NodeId) {
    this._nodes.delete(id);
  }

  /**
   * Removes the node that has the given `id` from this document and from
   * the given `layer`.
   *
   * Note: if the node belongs to another layer than the given `layer`,
   * then it will not be removed from that other layer, which will then still
   * reference the now-stale node.
   */
  removeNodeInLayer(id: NodeId, layer: Layer) {
    for (let i = layer.nodes.length - 1; i >= 0; i--) {
      if (layer.nodes[i] === id) {
        layer.nodes.splice(i, 1);
      }
    }
    this.removeNode(id);
  }

  // TODO: Safer / more convenient API that prevents layers having stale
  // nodes? For example, `layerId` could be a (readonly?) attribute of
  // each node, enforcing that each node only belongs to one layer, and
  // making it possible to automatically remove the node from its parent
  // layer in removeNode(). Also note that if instead of using the current
  // Node interface, client code were instead using some smarter Node
  // handle object that knows which document it belongs to, then it would be
  // possible to implement node.remove(), which may be an even more
  // convenient API, as close as possible as the UI equivalent of selecting
  // an node and deleting it via the delete key.

  /**
   * Finds the smallest positive integer `n` such that the name `${prefix}${n}`
   * is not taken by any of the given nodes, and returns that name.
   */
  findAvailableName(prefix: string, nodes: Array<NodeId>) {
    // Collect all positive integers from existing node names
    // that are of the form `${prefix}${number}`. Note that we need
    // the regex and not just rely on parseInt, since the latter
    // accepts +/-/e characters and stops at extra non-digit characters.
    const re = new RegExp(`${prefix}\\d+`);
    const numbers = [];
    for (const id of nodes) {
      const node = this.getNode(id);
      if (node && re.test(node.name)) {
        const suffix = node.name.substring(prefix.length);
        const n = parseInt(suffix, 10);
        if (!isNaN(n) && n > 0) {
          numbers.push(n);
        }
      }
    }
    // Find smallest available. This corresponds to the first
    // mismatch between `sorted` and [1, 2, 3, 4, ...].
    const sorted = sortAndRemoveDuplicates(numbers);
    let n = 1;
    for (const value of sorted) {
      if (value != n) {
        break;
      }
      n += 1;
    }
    return `${prefix}${n}`;
  }

  /**
   * Creates a new layer and add it to the document at the given index.
   *
   * If `index` is -1 (the default), the layer is added last.
   */
  createLayerAtIndex(index: number = -1): Document {
    const name = this.findAvailableName("Layer ", this.layers);
    const layer = this.createNode(Layer, { name: name });
    if (index < 0) {
      this.layers.push(layer.id);
    } else {
      this.layers.splice(index, 0, layer.id);
    }
    return this;
  }

  /**
   * Removes the layer at the given index.
   */
  deleteLayerAtIndex(index: number): Document {
    const id = this.layers[index];
    if (id === undefined) {
      return this;
    }
    const layer = this.getNode(id, Layer);
    if (layer === undefined) {
      return this;
    }
    this.layers.splice(index, 1);
    this._nodes.delete(id);
    return this;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                            Test Document

export function createTestDocument() {
  const doc = new Document();
  const layer = doc.createNode(Layer, { name: "Layer 1" });
  doc.layers = [layer.id];
  const p1 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(-100, 0),
  });
  const p2 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(0, 0),
  });
  const p3 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(100, 100),
  });
  const p4 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(200, 100),
  });
  const p5 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(200, 0),
  });
  const p6 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(100, -100),
  });
  const p7 = doc.createNodeInLayer(Point, layer, {
    position: new Vector2(-100, -100),
  });
  doc.createNodeInLayer(LineSegment, layer, {
    startPoint: p1.id,
    endPoint: p2.id,
  });
  doc.createNodeInLayer(ArcFromStartTangent, layer, {
    startPoint: p2.id,
    endPoint: p3.id,
    controlPoint: new Vector2(50, 0),
  });
  doc.createNodeInLayer(LineSegment, layer, {
    startPoint: p3.id,
    endPoint: p4.id,
  });
  doc.createNodeInLayer(CCurve, layer, {
    startPoint: p4.id,
    endPoint: p5.id,
    controlPoint: new Vector2(150, 50),
  });
  doc.createNodeInLayer(LineSegment, layer, {
    startPoint: p5.id,
    endPoint: p6.id,
  });
  doc.createNodeInLayer(SCurve, layer, {
    startPoint: p6.id,
    endPoint: p7.id,
    startControlPoint: new Vector2(50, -150),
    endControlPoint: new Vector2(-80, -60),
  });
  doc.createNodeInLayer(LineSegment, layer, {
    startPoint: p7.id,
    endPoint: p1.id,
  });
  doc.createNodeInLayer(LineSegment, layer, {
    startPoint: p2.id,
    endPoint: p6.id,
  });
  return doc;
}
