import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type NodeId = string;

export interface NodeOptions {
  readonly name?: string;
  readonly layer?: Layer | NodeId;
}

export abstract class Node {
  readonly doc: Document;
  readonly id: NodeId;
  readonly layer?: NodeId;
  name: string;

  constructor(doc: Document, id: NodeId, options: NodeOptions) {
    this.doc = doc;
    this.id = id;
    if (options.layer) {
      if (options.layer instanceof Layer) {
        this.layer = options.layer.id;
      } else {
        this.layer = options.layer;
      }
    }
    this.name = options.name !== undefined ? options.name : "Node";
  }

  abstract clone(newDoc: Document): Node;
}

/**
 * Represents any constructible Node type.
 *
 * This is used for typing factory functions like `Document.createNode()`.
 */
type NodeType<T, Options> = (new (
  doc: Document,
  id: NodeId,
  options: Options,
) => T) & {
  defaultName: string;
};

/**
 * Represents any Node type, not necessarily constructible.
 *
 * This is used for functions with runtime type checks like `Document.getNode()`.
 */
type AbstractNodeType<T> = abstract new (
  doc: Document,
  id: NodeId,
  options: never,
) => T;

///////////////////////////////////////////////////////////////////////////////
//                               Number

export interface NumberOptions extends NodeOptions {
  readonly value?: number;
}

export class Number extends Node {
  static readonly defaultName = "Number";
  value: number;

  constructor(doc: Document, id: NodeId, options: NumberOptions) {
    super(doc, id, options);
    this.value = options.value ? options.value : 0;
  }

  clone(newDoc: Document) {
    return new Number(newDoc, this.id, this);
  }

  static getOrCreate(
    doc: Document,
    id: NodeId | undefined,
    options: NumberOptions,
  ): NodeId {
    if (id === undefined) {
      return doc.createNode(Number, options).id;
    } else {
      // Note: if `id` is provided, then it is trusted to be in the document
      // and returned as is. This is important when cloning a whole document,
      // as the Number node may not be cloned yet, but another node being
      // cloned now may reference its ID.
      return id;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               SkeletonNode

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface SkeletonNodeOptions extends NodeOptions {}

export abstract class SkeletonNode extends Node {
  constructor(doc: Document, id: NodeId, options: SkeletonNodeOptions) {
    super(doc, id, options);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends SkeletonNodeOptions {
  readonly position?: Vector2 | [number, number];
  readonly x?: NodeId; // Number
  readonly y?: NodeId; // Number
}

function getVec2Pair(position?: Vector2 | [number, number]): [number, number] {
  if (!position) {
    return [0, 0];
  }
  if (position instanceof Vector2) {
    return [position.x, position.y];
  }
  return position;
}

export class Point extends SkeletonNode {
  static readonly defaultName = "Point";
  x: NodeId; // Number
  y: NodeId; // Number

  constructor(doc: Document, id: NodeId, options: PointOptions) {
    super(doc, id, options);
    const defaultPos = getVec2Pair(options.position);
    this.x = Number.getOrCreate(doc, options.x, {
      layer: options.layer,
      value: defaultPos[0],
    });
    this.y = Number.getOrCreate(doc, options.y, {
      layer: options.layer,
      value: defaultPos[1],
    });
  }

  clone(newDoc: Document) {
    return new Point(newDoc, this.id, this);
  }

  getPosition(): Vector2 {
    const x = this.doc.getNode(this.x, Number);
    const y = this.doc.getNode(this.y, Number);
    return new Vector2(x ? x.value : 0, y ? y.value : 0);
  }

  setPosition(position: Vector2) {
    const x = this.doc.getNode(this.x, Number);
    const y = this.doc.getNode(this.y, Number);
    if (x) {
      x.value = position.x;
    }
    if (y) {
      y.value = position.y;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               EdgeNode

export interface EdgeNodeOptions extends SkeletonNodeOptions {
  readonly startPoint: NodeId; // Point
  readonly endPoint: NodeId; // Point
}

export abstract class EdgeNode extends SkeletonNode {
  startPoint: NodeId; // Point
  endPoint: NodeId; // Point

  constructor(doc: Document, id: NodeId, options: EdgeNodeOptions) {
    super(doc, id, options);
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

  constructor(doc: Document, id: NodeId, options: LineSegmentOptions) {
    super(doc, id, options);
  }

  clone(newDoc: Document) {
    return new LineSegment(newDoc, this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               ArcFromStartTangent

export interface ArcFromStartTangentOptions extends EdgeNodeOptions {
  readonly controlPoint: NodeId; // Point
}

export class ArcFromStartTangent extends EdgeNode {
  static readonly defaultName = "Arc";
  controlPoint: NodeId; // Point

  constructor(doc: Document, id: NodeId, options: ArcFromStartTangentOptions) {
    super(doc, id, options);
    this.controlPoint = options.controlPoint;
  }

  clone(newDoc: Document) {
    return new ArcFromStartTangent(newDoc, this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  CCurve

export interface CCurveOptions extends EdgeNodeOptions {
  readonly controlPoint: NodeId; // Point
}

export class CCurve extends EdgeNode {
  static readonly defaultName = "C-Curve";
  controlPoint: NodeId; // Point

  constructor(doc: Document, id: NodeId, options: CCurveOptions) {
    super(doc, id, options);
    this.controlPoint = options.controlPoint;
  }

  clone(newDoc: Document) {
    return new CCurve(newDoc, this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  SCurve

export interface SCurveOptions extends EdgeNodeOptions {
  readonly startControlPoint: NodeId; // Point
  readonly endControlPoint: NodeId; // Point
}

export class SCurve extends EdgeNode {
  static readonly defaultName = "S-Curve";
  startControlPoint: NodeId; // Point
  endControlPoint: NodeId; // Point

  constructor(doc: Document, id: NodeId, options: SCurveOptions) {
    super(doc, id, options);
    this.startControlPoint = options.startControlPoint;
    this.endControlPoint = options.endControlPoint;
  }

  clone(newDoc: Document) {
    return new SCurve(newDoc, this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               MeasureNode

export interface MeasureNodeOptions extends NodeOptions {
  readonly isLocked?: boolean;
}

export abstract class MeasureNode extends Node {
  isLocked: boolean;

  constructor(doc: Document, id: NodeId, options: MeasureNodeOptions) {
    super(doc, id, options);
    this.isLocked = options.isLocked !== undefined ? options.isLocked : false;
  }

  abstract updateMeasure(): void;
}

///////////////////////////////////////////////////////////////////////////////
//                            PointToPointDistance

export interface PointToPointDistanceOptions extends MeasureNodeOptions {
  readonly startPoint: NodeId; // Point
  readonly endPoint: NodeId; // Point
  readonly value?: NodeId; // Number
}

export class PointToPointDistance extends MeasureNode {
  static readonly defaultName = "Point to Point Distance";
  startPoint: NodeId; // Point
  endPoint: NodeId; // Point
  value: NodeId; // Number

  constructor(doc: Document, id: NodeId, options: PointToPointDistanceOptions) {
    super(doc, id, options);
    this.startPoint = options.startPoint;
    this.endPoint = options.endPoint;

    this.value = Number.getOrCreate(doc, options.value, {
      layer: options.layer,
      value: 0,
    });
  }

  clone(newDoc: Document) {
    return new PointToPointDistance(newDoc, this.id, this);
  }

  updateMeasure() {
    if (this.isLocked) {
      return;
      // TODO: what to do if the measure is locked but the constraint
      // solver could not satisfy it? (overconstrained system) Shouldn't
      // we something show both the target value and the current value?
    }
    const p1 = this.doc.getNode(this.startPoint, Point);
    const p2 = this.doc.getNode(this.endPoint, Point);
    const v = this.doc.getNode(this.value, Number);
    if (!p1 || !p2 || !v) {
      return;
    }
    const value = p1.getPosition().distanceTo(p2.getPosition());
    v.value = value;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends NodeOptions {
  readonly nodes?: Array<NodeId>;
}

export class Layer extends Node {
  static readonly defaultName = "Layer";
  nodes: Array<NodeId>;

  constructor(doc: Document, id: NodeId, options: LayerOptions) {
    super(doc, id, options);
    this.nodes = options.nodes ? [...options.nodes] : [];
  }

  clone(newDoc: Document) {
    return new Layer(newDoc, this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneNodeMap(source: Map<NodeId, Node>, newDoc: Document) {
  const dest = new Map<NodeId, Node>();
  source.forEach((node, id) => {
    dest.set(id, node.clone(newDoc));
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
      this._nodes = cloneNodeMap(other._nodes, this);
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
   * Returns all the nodes in the document.
   */
  nodes(): MapIterator<Node> {
    return this._nodes.values();
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
   * If `options.layer` is provided, then the element is added to the layer.
   * Otherwise, it is created as a layer-less node.
   *
   * If `options.layer` is provided and `options.name` is not provided, then
   * this function will automatically assign a unique name within the `layer`
   * suitable for the given `spec`, e.g., "Point 42".
   */
  createNode<T extends Node, Options extends NodeOptions>(
    type: NodeType<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    let layer: undefined | Layer = undefined;
    if (options.layer instanceof Layer) {
      layer = options.layer;
    } else if (options.layer) {
      layer = this.getNode(options.layer, Layer);
    }
    if (layer && options.name === undefined) {
      const name = this.findAvailableName(`${type.defaultName} `, layer.nodes);
      options = { ...options, name: name };
    }
    const node = new type(this, id, options);
    this._nodes.set(id, node);
    if (layer) {
      layer.nodes.push(node.id);
    }
    if (node instanceof MeasureNode) {
      node.updateMeasure();
    }
    return node;
  }

  /**
   * Removes the node that has the given `id` from this document.
   */
  removeNode(id: NodeId) {
    const node = this.getNode(id);
    if (!node) {
      return;
    }
    if (node.layer !== undefined) {
      // Remove from layer
      const layer = this.getNode(node.layer, Layer);
      if (layer) {
        for (let i = layer.nodes.length - 1; i >= 0; i--) {
          if (layer.nodes[i] === id) {
            layer.nodes.splice(i, 1);
          }
        }
      }
    }
    this._nodes.delete(id);
  }

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
  const p1 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-100, 0),
  });
  const p2 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(0, 0),
  });
  const p3 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(100, 100),
  });
  const p4 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(200, 100),
  });
  const p5 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(200, 0),
  });
  const p6 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(100, -100),
  });
  const p7 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-100, -100),
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p1.id,
    endPoint: p2.id,
  });
  doc.createNode(PointToPointDistance, {
    layer: layer,
    startPoint: p1.id,
    endPoint: p2.id,
  });
  const cp1 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(50, 0),
  });
  doc.createNode(ArcFromStartTangent, {
    layer: layer,
    startPoint: p2.id,
    endPoint: p3.id,
    controlPoint: cp1.id,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p3.id,
    endPoint: p4.id,
  });
  const cp2 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(150, 50),
  });
  doc.createNode(CCurve, {
    layer: layer,
    startPoint: p4.id,
    endPoint: p5.id,
    controlPoint: cp2.id,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p5.id,
    endPoint: p6.id,
  });
  const cp3 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(50, -150),
  });
  const cp4 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-80, -60),
  });
  doc.createNode(SCurve, {
    layer: layer,
    startPoint: p6.id,
    endPoint: p7.id,
    startControlPoint: cp3.id,
    endControlPoint: cp4.id,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7.id,
    endPoint: p1.id,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2.id,
    endPoint: p6.id,
  });
  return doc;
}
