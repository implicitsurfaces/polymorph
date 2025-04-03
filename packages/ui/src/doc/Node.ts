import { Document } from "./Document";
import { Layer } from "./Layer";

/**
 * Reports a document error.
 */
export function logDocumentError(message: string) {
  console.log(`Document Error: ${message}`);
}

/**
 * Stores a unique ID of a given `Node` in a given `Document`.
 */
export type NodeId = string;

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
