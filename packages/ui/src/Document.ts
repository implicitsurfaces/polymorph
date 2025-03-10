import { Vector2 } from "threejs-math";
import { generate as generateShortUuId } from "short-uuid";

/**
 * Reports a document error.
 */
export function logDocumentError(message: string) {
  console.log(`Document Error: ${message}`);
}

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type NodeId = string;

// Note: We want the IDs to start with underscore and then only contain
// alphanumerics characters (no dashes) for use as is in the constraint
// solver. The default `short-uuid.generate()` (prefixed with underscore)
// satisfies this as Base58 encoding of UUID v4.
//
function generateNodeId(): NodeId {
  return `_${generateShortUuId()}`;
}

export interface NodeOptions {
  readonly name?: string;
  readonly layer?: Layer;
}

export abstract class Node {
  readonly doc: Document;
  readonly id: NodeId;

  private _layerId: NodeId | undefined;

  name: string;

  constructor(doc: Document, id: NodeId, options: NodeOptions) {
    this.doc = doc;
    this.id = id;
    if (options.layer) {
      this._layerId = options.layer.id;
    }
    this.name = options.name ?? "Node";
  }

  abstract clone(newDoc: Document): Node;

  get layer(): Layer | undefined {
    return this.doc.getNode(this._layerId, Layer);
  }

  /**
   * This helper function should be used in cases where it is expected
   * (e.g., method precondition) that an `id` corresponds to a given node
   * type. If not, an error is issued and `undefined` is returned, so the
   * caller has an opportunity to recover.
   */
  protected getExpectedNode<T extends Node>(
    id: NodeId | undefined,
    type: AbstractNodeType<T> | undefined,
  ): T | undefined {
    const node = this.doc.getNode(id, type);
    if (!node) {
      logDocumentError(
        `The given ID (${id}) does not correspond to a node of the expected type.`,
      );
    }
    return node;
  }

  /**
   * This helper function should be used in cases where a class invariant
   * enforces that an `id` must to be a given node type. If not, this
   * function throws.
   */
  protected getNodeAs<T extends Node>(
    id: NodeId | undefined,
    type: AbstractNodeType<T> | undefined,
  ): T {
    const node = this.doc.getNode(id, type);
    if (!node) {
      throw `The given ID (${id}) does not correspond to a node of the expected type.`;
    }
    return node;
  }
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

/**
 * This helper function returns the given `node` as is if not undefined,
 * otherwise it creates and returns a node of the given `type` and with the
 * given `options`.
 */
function getOrCreate<T extends Node, Options extends NodeOptions>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Options>,
  options: Options,
): T {
  if (node) {
    // Note: if `node` is given, then it is trusted to be in the document
    // and returned as is. This is important when cloning a whole document,
    // as the node may not be cloned yet, but another node being
    // cloned now may reference its ID.
    return node;
  } else {
    return doc.createNode(type, options);
  }
}

/**
 * This helper function is the same as `getOrCreate()` except that it returns
 * the node ID instead of the node itself.
 */
function getOrCreateId<T extends Node, Options extends NodeOptions>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Options>,
  options: Options,
): NodeId {
  return getOrCreate(doc, node, type, options).id;
}

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
}

///////////////////////////////////////////////////////////////////////////////
//                               SkeletonNode

export type SkeletonRole = "shape" | "construction";

export interface SkeletonNodeOptions extends NodeOptions {
  readonly role?: SkeletonRole;
}

export abstract class SkeletonNode extends Node {
  role: SkeletonRole;

  constructor(doc: Document, id: NodeId, options: SkeletonNodeOptions) {
    super(doc, id, options);
    this.role = options.role ?? "shape";
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends SkeletonNodeOptions {
  readonly position?: Vector2 | [number, number];
  readonly x?: Number;
  readonly y?: Number;
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
  private _xId: NodeId;
  private _yId: NodeId;

  constructor(doc: Document, id: NodeId, options: PointOptions) {
    super(doc, id, options);
    const defaultPos = getVec2Pair(options.position);
    this._xId = getOrCreateId(doc, options.x, Number, {
      layer: options.layer,
      value: defaultPos[0],
    });
    this._yId = getOrCreateId(doc, options.y, Number, {
      layer: options.layer,
      value: defaultPos[1],
    });
  }

  clone(newDoc: Document) {
    return new Point(newDoc, this.id, this);
  }

  get x(): Number {
    return this.getNodeAs(this._xId, Number);
  }

  set x(number: Number) {
    this._xId = number.id;
  }

  get y(): Number {
    return this.getNodeAs(this._yId, Number);
  }

  set y(number: Number) {
    this._yId = number.id;
  }

  get position(): Vector2 {
    return new Vector2(this.x.value, this.y.value);
  }

  set position(position: Vector2) {
    this.x.value = position.x;
    this.y.value = position.y;
  }

  // Note: For now, we do not provide ID-based getters and setters in order to
  // encourage the node-based interface and keep the interface smaller
  // (less boilerplate for each Node subtype). However, if we do want
  // ID-based getters/setters, they could look like the following:
  //
  // get xId(): NodeId {
  //   return this._xId;
  // }
  //
  // set xId(id: NodeId) {
  //   if (this.getExpectedNode(id, Number)) {
  //     this._xId = id;
  //   }
  // }
}

///////////////////////////////////////////////////////////////////////////////
//                               EdgeNode

export interface EdgeNodeOptions extends SkeletonNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
}

export abstract class EdgeNode extends SkeletonNode {
  private _startPointId: NodeId;
  private _endPointId: NodeId;

  constructor(doc: Document, id: NodeId, options: EdgeNodeOptions) {
    super(doc, id, options);
    this._startPointId = options.startPoint.id;
    this._endPointId = options.endPoint.id;
  }

  get startPoint(): Point {
    return this.getNodeAs(this._startPointId, Point);
  }

  set startPoint(point: Point) {
    this._startPointId = point.id;
  }

  get endPoint(): Point {
    return this.getNodeAs(this._endPointId, Point);
  }

  set endPoint(point: Point) {
    this._endPointId = point.id;
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
  readonly controlPoint: Point;
}

export class ArcFromStartTangent extends EdgeNode {
  static readonly defaultName = "Arc";
  private _controlPointId: NodeId;

  constructor(doc: Document, id: NodeId, options: ArcFromStartTangentOptions) {
    super(doc, id, options);
    this._controlPointId = options.controlPoint.id;
  }

  clone(newDoc: Document) {
    return new ArcFromStartTangent(newDoc, this.id, this);
  }

  get controlPoint(): Point {
    return this.getNodeAs(this._controlPointId, Point);
  }

  set controlPoint(point: Point) {
    this._controlPointId = point.id;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  CCurve

export interface CCurveOptions extends EdgeNodeOptions {
  readonly controlPoint: Point;
}

export class CCurve extends EdgeNode {
  static readonly defaultName = "C-Curve";
  private _controlPointId: NodeId;

  constructor(doc: Document, id: NodeId, options: CCurveOptions) {
    super(doc, id, options);
    this._controlPointId = options.controlPoint.id;
  }

  clone(newDoc: Document) {
    return new CCurve(newDoc, this.id, this);
  }

  get controlPoint(): Point {
    return this.getNodeAs(this._controlPointId, Point);
  }

  set controlPoint(point: Point) {
    this._controlPointId = point.id;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  SCurve

export interface SCurveOptions extends EdgeNodeOptions {
  readonly startControlPoint: Point;
  readonly endControlPoint: Point;
}

export class SCurve extends EdgeNode {
  static readonly defaultName = "S-Curve";
  private _startControlPointId: NodeId;
  private _endControlPointId: NodeId;

  constructor(doc: Document, id: NodeId, options: SCurveOptions) {
    super(doc, id, options);
    this._startControlPointId = options.startControlPoint.id;
    this._endControlPointId = options.endControlPoint.id;
  }

  clone(newDoc: Document) {
    return new SCurve(newDoc, this.id, this);
  }

  get startControlPoint(): Point {
    return this.getNodeAs(this._startControlPointId, Point);
  }

  set startControlPoint(point: Point) {
    this._startControlPointId = point.id;
  }

  get endControlPoint(): Point {
    return this.getNodeAs(this._endControlPointId, Point);
  }

  set endControlPoint(point: Point) {
    this._endControlPointId = point.id;
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
    this.isLocked = options.isLocked ?? true;
    // For now, we set the measure to be locked by default.
    // TODO: should the default be unlocked?
  }

  abstract updateMeasure(): void;
}

///////////////////////////////////////////////////////////////////////////////
//                            PointToPointDistance

export interface PointToPointDistanceOptions extends MeasureNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
  readonly number?: Number;
}

export class PointToPointDistance extends MeasureNode {
  static readonly defaultName = "Point to Point Distance";
  private _startPointId: NodeId;
  private _endPointId: NodeId;
  private _number: NodeId;

  constructor(doc: Document, id: NodeId, options: PointToPointDistanceOptions) {
    super(doc, id, options);
    this._startPointId = options.startPoint.id;
    this._endPointId = options.endPoint.id;
    this._number = getOrCreateId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
  }

  clone(newDoc: Document) {
    return new PointToPointDistance(newDoc, this.id, this);
  }

  get startPoint(): Point {
    return this.getNodeAs(this._startPointId, Point);
  }

  set startPoint(point: Point) {
    this._startPointId = point.id;
  }

  get endPoint(): Point {
    return this.getNodeAs(this._endPointId, Point);
  }

  set endPoint(point: Point) {
    this._endPointId = point.id;
  }

  get number(): Number {
    return this.getNodeAs(this._number, Number);
  }

  set number(number: Number) {
    this._number = number.id;
  }

  updateMeasure() {
    const startPosition = this.startPoint.position;
    const endPosition = this.endPoint.position;
    this.number.value = startPosition.distanceTo(endPosition);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends NodeOptions {
  readonly nodes?: NodeId[];
}

export class Layer extends Node {
  static readonly defaultName = "Layer";
  nodes: NodeId[];

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
  for (const [id, node] of source) {
    dest.set(id, node.clone(newDoc));
  }
  return dest;
}

function sortAndRemoveDuplicates(array: number[]) {
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

  public layers: NodeId[];

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
  getNodes<T extends Node>(ids: NodeId[], type?: AbstractNodeType<T>): T[] {
    const res: T[] = [];
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
    const id = generateNodeId();
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
   * Removes the given `node` from this document.
   */
  removeNode(node: Node) {
    if (node.doc != this) {
      return;
    }
    const layer = node.layer;
    if (layer) {
      // Remove from layer
      for (let i = layer.nodes.length - 1; i >= 0; i--) {
        if (layer.nodes[i] === node.id) {
          layer.nodes.splice(i, 1);
        }
      }
    }
    this._nodes.delete(node.id);
  }

  /**
   * Finds the smallest positive integer `n` such that the name `${prefix}${n}`
   * is not taken by any of the given nodes, and returns that name.
   */
  findAvailableName(prefix: string, nodes: NodeId[]) {
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
    startPoint: p1,
    endPoint: p2,
  });
  doc.createNode(PointToPointDistance, {
    layer: layer,
    startPoint: p1,
    endPoint: p2,
  });
  const cp1 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(50, 0),
  });
  doc.createNode(ArcFromStartTangent, {
    layer: layer,
    startPoint: p2,
    endPoint: p3,
    controlPoint: cp1,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p3,
    endPoint: p4,
  });
  const cp2 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(150, 50),
  });
  doc.createNode(CCurve, {
    layer: layer,
    startPoint: p4,
    endPoint: p5,
    controlPoint: cp2,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p5,
    endPoint: p6,
  });
  const cp3 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(50, -150),
  });
  const cp4 = doc.createNode(Point, {
    layer: layer,
    role: "construction",
    position: new Vector2(-80, -60),
  });
  doc.createNode(SCurve, {
    layer: layer,
    startPoint: p6,
    endPoint: p7,
    startControlPoint: cp3,
    endControlPoint: cp4,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7,
    endPoint: p1,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2,
    endPoint: p6,
  });
  return doc;
}
