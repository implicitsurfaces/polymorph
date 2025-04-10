import { NodeId } from "./NodeId";
import { AnyNodeData } from "./AnyNodeData";
import { asNodeIdArray, asBoolean } from "./dataFromAny";
import { log } from "./log";

// This file defines Node, Document, Layer, MeasureNode, and related types.
//
// For maintainability reasons, we would prefer defining them in their
// respective files (Node.ts, Document.ts, Layer.ts), but due to circular
// dependencies it is preferrable to define them in a single file. For
// example, circular dependencies tend to not work when used in a WebWorker.

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

  /**
   * Mutates the internal raw data stored in the `Node` to be the given
   * `data`.
   *
   * This assumes that the given `data` is indeed of the proper derived data
   * class, and therefore should be used with care. It it intended for use in
   * setters of abstract Node subclasses that would otherwise not be allowed
   * to mutate the internal data.
   */
  // TODO: better type-safe way to do this?
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
      log(
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
export type NodeConstructor<T, Data> = new (
  doc: Document,
  id: NodeId,
  data: Data,
) => T;

/**
 * The static variables and methods that any concrete node type must implement.
 */
export type NodeStatics<Data, Options> = {
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
export type AnyNodeType =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  NodeConstructor<Node, any> & NodeStatics<NodeData, any>;

/**
 * Stores meta-information about a concrete node type.
 */
export type NodeType<T, Data, Options> = NodeConstructor<T, Data> &
  NodeStatics<Data, Options>;

const _nodeRegistry: Record<string, AnyNodeType> = {};

/**
 * Adds a given node type to the registry. This is needed to be able to deserialize
 * JSON data whose type is only known from its type name.
 */
export function registerNodeType(nodeType: AnyNodeType) {
  _nodeRegistry[nodeType.name] = nodeType;
}

/**
 * Returns the registered node type, if any, corresponding to the given
 * node type name.
 */
export function getRegisteredNode(
  nodeTypeName: string,
): AnyNodeType | undefined {
  if (!(nodeTypeName in _nodeRegistry)) {
    return undefined;
  }
  return _nodeRegistry[nodeTypeName];
}

/**
 * Represents any Node type, not necessarily constructible.
 *
 * This is used for functions with runtime type checks like `Document.getNode()`.
 */
export type AbstractNodeType<T> = abstract new (
  doc: Document,
  id: NodeId,
  data: never,
) => T;

import { generate as generateShortUuId } from "short-uuid";

// Note: We want the IDs to start with underscore and then only contain
// alphanumerics characters (no dashes) for use as is in the constraint
// solver. The default `short-uuid.generate()` (prefixed with underscore)
// satisfies this as Base58 encoding of UUID v4.
//
function generateNodeId(): NodeId {
  return `_${generateShortUuId()}`;
}

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
    for (const rawNode of rawDoc.nodes) {
      const id = rawNode.id;
      const typename = rawNode.type;
      const type = getRegisteredNode(typename);
      if (!type) {
        throw Error(
          `Cannot parse Document from JSON: unknown node type ${typename}.`,
        );
      }
      const node: Node = new type(doc, id, type.dataFromAny(rawNode.data));
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

registerNodeType(Layer);

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
