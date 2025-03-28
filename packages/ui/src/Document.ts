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

/**
 * Stores a unique ID of a given `Node` in a given `Document`.
 */
export type NodeId = string;

// Note: We want the IDs to start with underscore and then only contain
// alphanumerics characters (no dashes) for use as is in the constraint
// solver. The default `short-uuid.generate()` (prefixed with underscore)
// satisfies this as Base58 encoding of UUID v4.
//
function generateNodeId(): NodeId {
  return `_${generateShortUuId()}`;
}

/**
 * Type of any `NodeData` property's value. This is essentially the same as a
 * parsed JSON value.
 */
export type AnyNodeDataValue =
  | string
  | number
  | boolean
  | null // [1]
  | readonly AnyNodeDataValue[]
  | AnyNodeData;

// [1] Unfortunately, we need to use `null` and not `undefined` for
// compatiblity with parsed JSON.
//
// This makes optional chaining uglier, for example `id = node?.id ?? null`
// instead of just `id = node?.id`, but it seems like a necessary evil.

/**
 * Type of any `NodeData`. This is essentially the same as a parsed JSON
 * object.
 */
export type AnyNodeData = { readonly [property: string]: AnyNodeDataValue };

// Note: we want node data to be immutable, and therefore arrays SHOULD be
// typed as `readonly T[]`:
//
// ```
// export interface CustomNodeData extends NodeData {
//   readonly nodeIds: readonly NodeId[];
// }
// ```
//
// This is why we need `readonly` in the `AnyNodeData` and `AnyNodeDataValue`
// type definitions, otherwise the snippet above would produce an error, because
// `readonly T[]` cannot be assigned to type `T[]`.
//
// Unfortunately, the following currently does not produce an error, but you
// SHOULD NOT do it:
//
// ```
// export interface CustomNodeData extends NodeData {
//   readonly nodeIds: NodeId[];
// }
// ```
//
// If the future we find a way to make the above an error, we will.
//
// The reason it does not currently produce an error is that `T[]` can be
// assigned to type `readonly T[]`, and therefore the type `{ foo: T
// [] }` does extend `{ foo: readonly T[]; }`.

/**
 * Data shared by all `Node` subclasses.
 *
 * This class extends `AnyNodeData` to enforce that any of its subclasses is
 * JSON-like. For example, the following is an error:
 *
 * ```
 * export interface CustomNodeData extends NodeData {
 *   readonly otherNode: Node;
 * }
 * ```
 *
 * Error message:
 *
 * ```
 * Property 'otherNode' of type 'Node' is not assignable to 'string' index
 * type 'AnyNodeDataValue'.
 * ```
 */
export interface NodeData extends AnyNodeData {
  readonly name: string;
  readonly layerId: NodeId | null;
}

/**
 * Options shared by all `Node` subclasses.
 */
export interface NodeOptions {
  readonly name?: string;
  readonly layer?: Layer;
}

/**
 * Abstract base class for all node types in a `Document`.
 */
export abstract class Node {
  constructor(
    readonly doc: Document,
    readonly id: NodeId,
  ) {}

  clone(newDoc: Document): Node {
    return new (this.constructor as NodeConstructor<Node, NodeData>)(
      newDoc,
      this.id,
      this.data,
    );
  }

  /**
   * Returns the raw data stored in the `Node`.
   *
   * Note that this data is usually not very convenient to work with, as it
   * often contains `NodeId` properties rather than actual `Node` objects.
   *
   * For this reason, it is often more convenient to use instead the other
   * getters, for example `node.layer` returns a `Layer` which is typically
   * more useful that `node.data.layerId`.
   *
   * This method is abstract because it is the responsibility of each concrete
   * `Node` subclass to store its data with its more specific type.
   */
  abstract get data(): NodeData;

  protected abstract setData(data: NodeData): void;

  /**
   * Converts convenient (and possibly optional) parameters into immutable
   * type-checked node data, in the context of the given document.
   *
   * Warning: in some subclasses, this function may mutate the given
   * document!
   *
   * For example, `Point.dataFromOptions(doc, { position: [x, y] })`
   * automatically creates two `Number` nodes in the document to store the
   * coordinates of point's position.
   */
  static dataFromOptions(_doc: Document, options: NodeOptions): NodeData {
    return {
      name: options.name ?? "Node",
      layerId: options.layer?.id ?? null,
    };
  }

  /**
   * Converts generic node data (for example, obtained by parsing a JSON
   * string) into immutable type-checked node data.
   */
  static dataFromAny(d: AnyNodeData): NodeData {
    return {
      name: typeof d.name === "string" ? d.name : "Node",
      layerId: typeof d.layerId === "string" ? d.layerId : null,
    };
  }

  /**
   * Returns a JSON representation of this `Node`, which includes its type,
   * ID, and raw data.
   */
  toJSON(): string {
    return JSON.stringify({
      type: this.type,
      id: this.id,
      data: this.data,
    });

    // Note 1: `doc` is intentionally omitted as it is essentially only a
    // convenient "back pointer", which means that:
    //
    // 1. it is not necessary for deserialization, and
    //
    // 2. it would lead to an infinite recursion in stringify.
    //
    // Note 2: For now, `layerID` is not omitted, although it could be as it
    // can also be seen as a back pointer. However, a practical way to store
    // a tree in a concurrent multi-user system is actually a flat unordered
    // list of nodes where each node stores its parent and fractional index,
    // rather than having each node store its children as a list. The tree
    // can then be reconstructed locally for UI presentation and
    // manipulation.
  }

  /**
   * Returns the name of this node's type.
   *
   * Example: `"Point"`.
   *
   * This is equivalent to `this.constructor.name`.
   */
  get type(): string {
    return this.constructor.name;
  }

  /**
   * Returns the user-visible name of this specific node instance.
   *
   * Example: `"Point 42"`.
   */
  get name(): string {
    return this.data.name;
  }

  /**
   * Returns which layer this node belongs to, if any.
   */
  get layer(): Layer | undefined {
    return this.doc.getNode(this.data.layerId, Layer);
  }

  /**
   * This helper function should be used in cases where it is expected
   * (e.g., method precondition) that the given `id` corresponds to a node of
   * the given `type`. If not, an error is issued and `undefined` is
   * returned, so the caller has an opportunity to recover.
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
   * enforces that the given `id` corresponds to a node of the given `type`.
   * If not, this function throws.
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
 * The signature of the constructor that any concrete node type must have.
 */
//type AnyNodeConstructor = new (doc: Document, id: NodeId, data: any) => Node;

/**
 * The signature of the constructor that any concrete node type must have.
 */
type NodeConstructor<T, Data> = new (
  doc: Document,
  id: NodeId,
  data: Data,
) => T;

/**
 * The static variables and methods that any concrete node type must implement.
 */
type NodeStatics<Data, Options> = {
  /**
   * The unique name of the node type that can be used as key.
   *
   * This is automatically generated by JavaScript, for example, for the
   * `Point` class, this is `Point.constructor.name`.
   **/
  name: string;

  /**
   * The default user-visible name that is given to instances of this node
   * type.
   *
   * For example, if defaultName is "Point", then by default instances of this
   * node type will be named "Point 1", "Point 2", etc.
   **/
  defaultName: string;

  /**
   * Creates type-checked data from generic data.
   **/
  dataFromAny(d: AnyNodeData): Data;

  /**
   * Creates type-checked data from options.
   */
  dataFromOptions(doc: Document, options: Options): Data;
};

/**
 * Stores meta-information about any concrete node type.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNodeType = NodeConstructor<Node, any> & NodeStatics<NodeData, any>;

/**
 * Stores meta-information about a concrete node type.
 */
type NodeType<T, Data, Options> = NodeConstructor<T, Data> &
  NodeStatics<Data, Options>;

const NODE_REGISTRY: Record<string, AnyNodeType> = {};

/**
 * Adds a given node type to the registry. This is needed to be able to deserialize
 * JSON data whose type is only known from its type name.
 */
function registerNode(nodeType: AnyNodeType) {
  NODE_REGISTRY[nodeType.name] = nodeType;
}

/**
 * Represents any Node type, not necessarily constructible.
 *
 * This is used for functions with runtime type checks like `Document.getNode()`.
 */
type AbstractNodeType<T> = abstract new (
  doc: Document,
  id: NodeId,
  data: never,
) => T;

/**
 * This helper function returns the given `node` as is if not undefined,
 * otherwise it creates and returns a node of the given `type` and with the
 * given `options`.
 */
function getOrCreate<
  T extends Node,
  Data extends NodeData,
  Options extends NodeOptions,
>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Data, Options>,
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
function getOrCreateId<
  T extends Node,
  Data extends NodeData,
  Options extends NodeOptions,
>(
  doc: Document,
  node: T | undefined,
  type: NodeType<T, Data, Options>,
  options: Options,
): NodeId {
  return getOrCreate(doc, node, type, options).id;
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type string.
 *
 * If it does, then this function returns the string as a `NodeId`.
 *
 * Otherwise, this function throws.
 */
function asNodeId(data: AnyNodeData, property: string): NodeId {
  if (property in data) {
    const id = data[property];
    if (typeof id === "string") {
      return id;
    } else {
      throw Error(
        `ID property "${property}" is not of type string in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing ID property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type boolean.
 *
 * If it does, then this function returns the boolean property value.
 *
 * Otherwise, this function throws.
 */
function asBoolean(data: AnyNodeData, property: string): boolean {
  if (property in data) {
    const b = data[property];
    if (typeof b === "boolean") {
      return b;
    } else {
      throw Error(
        `Property "${property}" is not of type boolean in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing boolean property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type string array.
 *
 * If it does, then this function returns the array as `NodeId[]`.
 *
 * Otherwise, this function throws.
 */
function asNodeIdArray(data: AnyNodeData, property: string): NodeId[] {
  if (property in data) {
    const a = data[property];
    if (Array.isArray(a)) {
      for (const id of a) {
        if (typeof id != "string") {
          throw Error(
            `Found non-ID value in property "${property}" in the given data (=${data}).`,
          );
        }
      }
      return a;
    } else {
      throw Error(
        `Property "${property}" is not of type array of IDs in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing array of IDs property "${property}" in the given data (=${data}).`,
    );
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Number

export interface NumberData extends NodeData {
  readonly value: number;
}

export interface NumberOptions extends NodeOptions {
  readonly value?: number;
}

export class Number extends Node {
  static readonly defaultName = "Number";

  private _data: NumberData;

  get data(): NumberData {
    return this._data;
  }

  setData(data: NumberData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: NumberData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: NumberOptions): NumberData {
    return {
      ...Node.dataFromOptions(doc, options),
      value: options.value ?? 0,
    };
  }

  static dataFromAny(d: AnyNodeData): NumberData {
    return {
      ...Node.dataFromAny(d),
      value: typeof d.value === "number" ? d.value : 0,
    };
  }

  get value(): number {
    return this.data.value;
  }

  set value(newValue: number) {
    this._data = {
      ...this.data,
      value: newValue,
    };
  }
}

registerNode(Number);

///////////////////////////////////////////////////////////////////////////////
//                               SkeletonNode

export type SkeletonRole = "shape" | "construction";

function isSkeletonRole(v: unknown): v is SkeletonRole {
  return v === "shape" || v === "construction";
}

export interface SkeletonNodeData extends NodeData {
  readonly role: SkeletonRole;
}

export interface SkeletonNodeOptions extends NodeOptions {
  readonly role?: SkeletonRole;
}

export abstract class SkeletonNode extends Node {
  abstract get data(): SkeletonNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: SkeletonNodeOptions,
  ): SkeletonNodeData {
    return {
      ...Node.dataFromOptions(doc, options),
      role: options.role ?? "shape",
    };
  }

  static dataFromAny(d: AnyNodeData): SkeletonNodeData {
    return {
      ...Node.dataFromAny(d),
      role: isSkeletonRole(d.role) ? d.role : "shape",
    };
  }

  get role(): SkeletonRole {
    return this.data.role;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointData extends SkeletonNodeData {
  readonly xId: NodeId;
  readonly yId: NodeId;
}

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

  private _data: PointData;

  get data(): PointData {
    return this._data;
  }

  setData(data: PointData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: PointData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: PointOptions): PointData {
    const defaultPos = getVec2Pair(options.position);
    const xId = getOrCreateId(doc, options.x, Number, {
      layer: options.layer,
      value: defaultPos[0],
    });
    const yId = getOrCreateId(doc, options.y, Number, {
      layer: options.layer,
      value: defaultPos[1],
    });
    return {
      ...SkeletonNode.dataFromOptions(doc, options),
      xId: xId,
      yId: yId,
    };
  }

  static dataFromAny(d: AnyNodeData): PointData {
    return {
      ...SkeletonNode.dataFromAny(d),
      xId: asNodeId(d, "xId"),
      yId: asNodeId(d, "yId"),
    };
  }

  get x(): Number {
    return this.getNodeAs(this.data.xId, Number);
  }

  set x(number: Number) {
    this._data = {
      ...this.data,
      xId: number.id,
    };
  }

  get y(): Number {
    return this.getNodeAs(this.data.yId, Number);
  }

  set y(number: Number) {
    this._data = {
      ...this.data,
      yId: number.id,
    };
  }

  get position(): Vector2 {
    return new Vector2(this.x.value, this.y.value);
  }

  set position(position: Vector2) {
    this.x.value = position.x;
    this.y.value = position.y;
  }
}

registerNode(Point);

///////////////////////////////////////////////////////////////////////////////
//                               EdgeNode

export interface EdgeNodeData extends SkeletonNodeData {
  readonly startPointId: NodeId;
  readonly endPointId: NodeId;
}

export interface EdgeNodeOptions extends SkeletonNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
}

export abstract class EdgeNode extends SkeletonNode {
  abstract get data(): EdgeNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: EdgeNodeOptions,
  ): EdgeNodeData {
    return {
      ...SkeletonNode.dataFromOptions(doc, options),
      startPointId: options.startPoint.id,
      endPointId: options.endPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): EdgeNodeData {
    return {
      ...SkeletonNode.dataFromAny(d),
      startPointId: asNodeId(d, "startPointId"),
      endPointId: asNodeId(d, "endPointId"),
    };
  }

  get startPoint(): Point {
    return this.getNodeAs(this.data.startPointId, Point);
  }

  set startPoint(point: Point) {
    this.setData({
      ...this.data,
      startPointId: point.id,
    });
  }

  get endPoint(): Point {
    return this.getNodeAs(this.data.endPointId, Point);
  }

  set endPoint(point: Point) {
    this.setData({
      ...this.data,
      endPointId: point.id,
    });
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               LineSegment

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentData extends EdgeNodeData {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeNodeOptions {}

export class LineSegment extends EdgeNode {
  static readonly defaultName = "Line Segment";

  private _data: LineSegmentData;

  get data(): LineSegmentData {
    return this._data;
  }

  setData(data: LineSegmentData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LineSegmentData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: LineSegmentOptions,
  ): LineSegmentData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
    };
  }

  static dataFromAny(d: AnyNodeData): LineSegmentData {
    return {
      ...EdgeNode.dataFromAny(d),
    };
  }
}

registerNode(LineSegment);

///////////////////////////////////////////////////////////////////////////////
//                               ArcFromStartTangent

export interface ArcFromStartTangentData extends EdgeNodeData {
  readonly controlPointId: NodeId;
}

export interface ArcFromStartTangentOptions extends EdgeNodeOptions {
  readonly controlPoint: Point;
}

export class ArcFromStartTangent extends EdgeNode {
  static readonly defaultName = "Arc";

  private _data: ArcFromStartTangentData;

  get data(): ArcFromStartTangentData {
    return this._data;
  }

  setData(data: ArcFromStartTangentData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: ArcFromStartTangentData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: ArcFromStartTangentOptions,
  ): ArcFromStartTangentData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      controlPointId: options.controlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): ArcFromStartTangentData {
    return {
      ...EdgeNode.dataFromAny(d),
      controlPointId: asNodeId(d, "controlPointId"),
    };
  }

  get controlPoint(): Point {
    return this.getNodeAs(this.data.controlPointId, Point);
  }

  set controlPoint(point: Point) {
    this._data = {
      ...this.data,
      controlPointId: point.id,
    };
  }
}

registerNode(ArcFromStartTangent);

///////////////////////////////////////////////////////////////////////////////
//                                  CCurve

export interface CCurveData extends EdgeNodeData {
  readonly controlPointId: NodeId;
}

export interface CCurveOptions extends EdgeNodeOptions {
  readonly controlPoint: Point;
}

export class CCurve extends EdgeNode {
  static readonly defaultName = "C-Curve";

  private _data: CCurveData;

  get data(): CCurveData {
    return this._data;
  }

  setData(data: CCurveData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: CCurveData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: CCurveOptions): CCurveData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      controlPointId: options.controlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): CCurveData {
    return {
      ...EdgeNode.dataFromAny(d),
      controlPointId: asNodeId(d, "controlPointId"),
    };
  }

  get controlPoint(): Point {
    return this.getNodeAs(this.data.controlPointId, Point);
  }

  set controlPoint(point: Point) {
    this._data = {
      ...this.data,
      controlPointId: point.id,
    };
  }
}

registerNode(CCurve);

///////////////////////////////////////////////////////////////////////////////
//                                  SCurve

export interface SCurveData extends EdgeNodeData {
  readonly startControlPointId: NodeId;
  readonly endControlPointId: NodeId;
}

export interface SCurveOptions extends EdgeNodeOptions {
  readonly startControlPoint: Point;
  readonly endControlPoint: Point;
}

export class SCurve extends EdgeNode {
  static readonly defaultName = "S-Curve";

  private _data: SCurveData;

  get data(): SCurveData {
    return this._data;
  }

  setData(data: SCurveData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: SCurveData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: SCurveOptions): SCurveData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      startControlPointId: options.startControlPoint.id,
      endControlPointId: options.endControlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): SCurveData {
    return {
      ...EdgeNode.dataFromAny(d),
      startControlPointId: asNodeId(d, "startControlPointId"),
      endControlPointId: asNodeId(d, "endControlPointId"),
    };
  }

  get startControlPoint(): Point {
    return this.getNodeAs(this.data.startControlPointId, Point);
  }

  set startControlPoint(point: Point) {
    this._data = {
      ...this.data,
      startControlPointId: point.id,
    };
  }

  get endControlPoint(): Point {
    return this.getNodeAs(this.data.endControlPointId, Point);
  }

  set endControlPoint(point: Point) {
    this._data = {
      ...this.data,
      endControlPointId: point.id,
    };
  }
}

registerNode(SCurve);

///////////////////////////////////////////////////////////////////////////////
//                               MeasureNode

export interface MeasureNodeData extends NodeData {
  readonly isLocked: boolean;
}

export interface MeasureNodeOptions extends NodeOptions {
  readonly isLocked?: boolean;
}

export abstract class MeasureNode extends Node {
  abstract get data(): MeasureNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: MeasureNodeOptions,
  ): MeasureNodeData {
    return {
      ...Node.dataFromOptions(doc, options),
      isLocked: options.isLocked ?? true,
    };
    // For now, we set the measure to be locked by default.
    // TODO: should the default be unlocked?
  }

  static dataFromAny(d: AnyNodeData): MeasureNodeData {
    return {
      ...Node.dataFromAny(d),
      isLocked: asBoolean(d, "isLocked"),
    };
  }

  abstract updateMeasure(): void;

  get isLocked(): boolean {
    return this.data.isLocked;
  }

  set isLocked(locked: boolean) {
    this.setData({
      ...this.data,
      isLocked: locked,
    });
  }
}

///////////////////////////////////////////////////////////////////////////////
//                            PointToPointDistance

export interface PointToPointDistanceData extends MeasureNodeData {
  readonly startPointId: NodeId;
  readonly endPointId: NodeId;
  readonly numberId: NodeId;
}

export interface PointToPointDistanceOptions extends MeasureNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
  readonly number?: Number;
}

export class PointToPointDistance extends MeasureNode {
  static readonly defaultName = "Point to Point Distance";

  private _data: PointToPointDistanceData;

  get data(): PointToPointDistanceData {
    return this._data;
  }

  setData(data: PointToPointDistanceData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: PointToPointDistanceData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: PointToPointDistanceOptions,
  ): PointToPointDistanceData {
    const numberId = getOrCreateId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      startPointId: options.startPoint.id,
      endPointId: options.endPoint.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): PointToPointDistanceData {
    return {
      ...MeasureNode.dataFromAny(d),
      startPointId: asNodeId(d, "startPointId"),
      endPointId: asNodeId(d, "endPointId"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get startPoint(): Point {
    return this.getNodeAs(this.data.startPointId, Point);
  }

  set startPoint(point: Point) {
    this._data = {
      ...this.data,
      startPointId: point.id,
    };
  }

  get endPoint(): Point {
    return this.getNodeAs(this.data.endPointId, Point);
  }

  set endPoint(point: Point) {
    this._data = {
      ...this.data,
      endPointId: point.id,
    };
  }

  get number(): Number {
    return this.getNodeAs(this.data.numberId, Number);
  }

  set number(number: Number) {
    this._data = {
      ...this.data,
      numberId: number.id,
    };
  }

  updateMeasure() {
    const startPosition = this.startPoint.position;
    const endPosition = this.endPoint.position;
    this.number.value = startPosition.distanceTo(endPosition);
  }
}

registerNode(PointToPointDistance);

///////////////////////////////////////////////////////////////////////////////
//                            LineToPointDistance

export interface LineToPointDistanceData extends MeasureNodeData {
  readonly lineId: NodeId;
  readonly pointId: NodeId;
  readonly numberId: NodeId;
}

export interface LineToPointDistanceOptions extends MeasureNodeOptions {
  readonly line: LineSegment;
  readonly point: Point;
  readonly number?: Number;
}

export class LineToPointDistance extends MeasureNode {
  static readonly defaultName = "Line to Point Distance";

  private _data: LineToPointDistanceData;

  get data(): LineToPointDistanceData {
    return this._data;
  }

  setData(data: LineToPointDistanceData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LineToPointDistanceData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: LineToPointDistanceOptions,
  ): LineToPointDistanceData {
    const numberId = getOrCreateId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      lineId: options.line.id,
      pointId: options.point.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): LineToPointDistanceData {
    return {
      ...MeasureNode.dataFromAny(d),
      lineId: asNodeId(d, "lineId"),
      pointId: asNodeId(d, "pointId"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get line(): LineSegment {
    return this.getNodeAs(this.data.lineId, LineSegment);
  }

  set line(line: LineSegment) {
    this._data = {
      ...this.data,
      lineId: line.id,
    };
  }

  get point(): Point {
    return this.getNodeAs(this.data.pointId, Point);
  }

  set point(point: Point) {
    this._data = {
      ...this.data,
      startPointId: point.id,
    };
  }

  get number(): Number {
    return this.getNodeAs(this.data.numberId, Number);
  }

  set number(number: Number) {
    this._data = {
      ...this.data,
      numberId: number.id,
    };
  }

  updateMeasure() {
    const p1x = this.line.startPoint.position.x;
    const p1y = this.line.startPoint.position.y;
    const p2x = this.line.endPoint.position.x;
    const p2y = this.line.endPoint.position.y;
    const px = this.point.position.x;
    const py = this.point.position.y;

    // Calculate distance from point to infinite line
    // using formula |ax + by + c| / sqrt(a^2 + b^2)
    // where ax + by + c = 0 is line equation
    const a = p2y - p1y;
    const b = p1x - p2x;
    const c = p2x * p1y - p1x * p2y;

    const distance = Math.abs(a * px + b * py + c) / Math.sqrt(a * a + b * b);

    this.number.value = distance;
  }
}

registerNode(LineToPointDistance);

///////////////////////////////////////////////////////////////////////////////
//                            Angle

export interface AngleData extends MeasureNodeData {
  readonly line0Id: NodeId;
  readonly line1Id: NodeId;
  readonly numberId: NodeId;
}

export interface AngleOptions extends MeasureNodeOptions {
  readonly line0: LineSegment;
  readonly line1: Point;
  readonly number?: Number;
}

export class Angle extends MeasureNode {
  static readonly defaultName = "Line to Line Angle";

  private _data: AngleData;

  get data(): AngleData {
    return this._data;
  }

  setData(data: AngleData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: AngleData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: AngleOptions): AngleData {
    const numberId = getOrCreateId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      line0Id: options.line0.id,
      line1Id: options.line1.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): AngleData {
    return {
      ...MeasureNode.dataFromAny(d),
      line0Id: asNodeId(d, "line0Id"),
      line1Id: asNodeId(d, "line1Id"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get line0(): LineSegment {
    return this.getNodeAs(this.data.line0Id, LineSegment);
  }

  set line0(line: LineSegment) {
    this._data = {
      ...this.data,
      line0Id: line.id,
    };
  }

  get line1(): LineSegment {
    return this.getNodeAs(this.data.line1Id, LineSegment);
  }

  set line1(line: LineSegment) {
    this._data = {
      ...this.data,
      line1Id: line.id,
    };
  }

  get number(): Number {
    return this.getNodeAs(this.data.numberId, Number);
  }

  set number(number: Number) {
    this._data = {
      ...this.data,
      numberId: number.id,
    };
  }

  updateMeasure() {
    const p1x = this.line0.startPoint.position.x;
    const p1y = this.line0.startPoint.position.y;
    const p2x = this.line0.endPoint.position.x;
    const p2y = this.line0.endPoint.position.y;
    const p3x = this.line1.startPoint.position.x;
    const p3y = this.line1.startPoint.position.y;
    const p4x = this.line1.endPoint.position.x;
    const p4y = this.line1.endPoint.position.y;

    // Calculate vectors for both lines
    const v1x = p2x - p1x;
    const v1y = p2y - p1y;
    const v2x = p4x - p3x;
    const v2y = p4y - p3y;

    // Calculate angle between vectors using dot product
    const dot = v1x * v2x + v1y * v2y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);

    let angle = Math.acos(dot / (mag1 * mag2));

    // Convert to degrees
    angle = (angle * 180) / Math.PI;

    // Normalize to 0-180 range
    angle = angle > 180 ? angle - 180 : angle;

    this.number.value = angle;
  }
}

registerNode(Angle);

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerData extends NodeData {
  readonly nodeIds: readonly NodeId[];
}

export interface LayerOptions extends NodeOptions {
  readonly nodes?: readonly Node[];
}

export class Layer extends Node {
  static readonly defaultName = "Layer";

  private _data: LayerData;

  get data(): LayerData {
    return this._data;
  }

  setData(data: LayerData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LayerData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: LayerOptions): LayerData {
    return {
      ...Node.dataFromOptions(doc, options),
      nodeIds: options.nodes?.map((node) => node.id) ?? [],
    };
  }

  static dataFromAny(d: AnyNodeData): LayerData {
    return {
      ...Node.dataFromAny(d),
      nodeIds: asNodeIdArray(d, "nodeIds"),
    };
  }

  get nodes(): Node[] {
    return this.data.nodeIds.map((id) => this.getNodeAs(id, Node));
  }

  addNode(node: Node) {
    this._data = {
      ...this.data,
      nodeIds: [...this.data.nodeIds, node.id],
    };
  }

  removeNode(node: Node) {
    const newNodeIds: NodeId[] = [];
    for (const id of this.data.nodeIds) {
      if (id !== node.id) {
        newNodeIds.push(node.id);
      }
    }
    this._data = {
      ...this.data,
      nodeIds: newNodeIds,
    };
  }
}

registerNode(Layer);

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
   * Serializes the Document into JSON.
   */
  // Note: do not rely on this specific JSON structure, we might change it in the future.
  //
  toJSON(): string {
    return JSON.stringify({
      layers: this.layers,
      nodes: Array.from(this._nodes.values()).map((node) => {
        return { type: node.type, id: node.id, data: node.data };
      }),
    });
  }

  /**
   * Create a Document from JSON.
   */
  static fromJSON(json: string): Document {
    const rawDoc = JSON.parse(json);
    const doc = new Document();
    doc.layers = rawDoc.layers;
    console.log("Layers set to: ", doc.layers);
    for (const rawNode of rawDoc.nodes) {
      console.log("Processing node:");
      console.log(rawNode);
      const id = rawNode.id;
      const typename = rawNode.type;
      if (!(typename in NODE_REGISTRY)) {
        throw Error(
          `Cannot parse Document from JSON: unknown node type ${typename}.`,
        );
      }
      const type = NODE_REGISTRY[typename];
      const node: Node = new type(doc, id, type.dataFromAny(rawNode.data));
      console.log("Resulting node:");
      console.log(node);
      doc._nodes.set(id, node);
    }
    return doc;
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
    id: NodeId | undefined | null,
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
    ids: readonly NodeId[],
    type?: AbstractNodeType<T>,
  ): T[] {
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
  createNode<
    T extends Node,
    Data extends NodeData,
    Options extends NodeOptions,
  >(type: NodeType<T, Data, Options>, options: Options): T {
    const id = generateNodeId();
    let layer: undefined | Layer = undefined;
    if (options.layer instanceof Layer) {
      layer = options.layer;
    } else if (options.layer) {
      layer = this.getNode(options.layer, Layer);
    }
    if (layer && options.name === undefined) {
      const name = this.findAvailableName(
        `${type.defaultName} `,
        layer.data.nodeIds,
      );
      options = { ...options, name: name };
    }
    const data = type.dataFromOptions(this, options);
    const node = new type(this, id, data);
    this._nodes.set(id, node);
    if (layer) {
      layer.addNode(node);
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
      layer.removeNode(node);
    }
    this._nodes.delete(node.id);
  }

  /**
   * Finds the smallest positive integer `n` such that the name `${prefix}${n}`
   * is not taken by any of the given nodes, and returns that name.
   */
  findAvailableName(prefix: string, nodes: readonly NodeId[]) {
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

/*
export function createTestDocument_v0() {
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
  const l1 = doc.createNode(LineSegment, {
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
  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l1,
    point: cp1,
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
  const l2 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7,
    endPoint: p1,
  });
  doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2,
    endPoint: p6,
  });
  doc.createNode(Angle, {
    layer: layer,
    line0: l2,
    line1: l1,
  });

  console.log(doc);
  return doc;
}
*/

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
  const l1 = doc.createNode(LineSegment, {
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
    position: new Vector2(50, 0),
  });
  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l1,
    point: cp1,
  });
  doc.createNode(ArcFromStartTangent, {
    layer: layer,
    startPoint: p2,
    endPoint: p3,
    controlPoint: cp1,
  });
  const l4 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p3,
    endPoint: p4,
  });
  const cp2 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(320, 100),
  });
  doc.createNode(CCurve, {
    layer: layer,
    startPoint: p4,
    endPoint: p5,
    controlPoint: cp2,
  });
  const l5 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p5,
    endPoint: p6,
  });
  const cp3 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(50, -150),
  });
  const cp4 = doc.createNode(Point, {
    layer: layer,
    position: new Vector2(-100, -150),
  });
  doc.createNode(SCurve, {
    layer: layer,
    startPoint: p6,
    endPoint: p7,
    startControlPoint: cp3,
    endControlPoint: cp4,
  });
  const l2 = doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p7,
    endPoint: p1,
  });
  /*const l3 =*/ doc.createNode(LineSegment, {
    layer: layer,
    startPoint: p2,
    endPoint: p6,
  });

  // doc.createNode(Angle, {
  //   layer: layer,
  //   line0: l2,
  //   line1: l1,
  // });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l4,
    point: cp2,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l5,
    point: cp2,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l2,
    point: cp4,
  });

  doc.createNode(LineToPointDistance, {
    layer: layer,
    line: l5,
    point: cp3,
  });

  console.log(doc);
  console.log(doc.toJSON());

  const json = doc.toJSON();

  const doc2 = Document.fromJSON(json);
  return doc2;
}
