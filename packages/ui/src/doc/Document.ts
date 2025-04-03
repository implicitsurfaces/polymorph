import { generate as generateShortUuId } from "short-uuid";

import {
  AbstractNodeType,
  Node,
  NodeData,
  NodeId,
  NodeOptions,
  NodeType,
  getRegisteredNode,
} from "./Node";

import { MeasureNode } from "./MeasureNode";
import { Layer } from "./Layer";

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
