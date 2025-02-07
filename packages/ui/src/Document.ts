import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type ElementId = string;

export interface ElementOptions {
  name?: string;
}

export abstract class Element {
  readonly id: ElementId;
  name: string;

  constructor(id: ElementId, options: ElementOptions) {
    this.id = id;
    this.name = options.name !== undefined ? options.name : "Element";
  }

  abstract clone(): Element;
}

/**
 * Represents any constructible Element type.
 *
 * This is used for typing factory functions like `Document.createElement()`.
 */
type ElementType<T, Options> = (new (id: ElementId, options: Options) => T) & {
  defaultName: string;
};

/**
 * Represents any Element type, not necessarily constructible.
 *
 * This is used for functions with runtime type checks like `Document.getElement()`.
 */
type AbstractElementType<T> = abstract new (id: ElementId, options: never) => T;

///////////////////////////////////////////////////////////////////////////////
//                               SkeletonElement

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface SkeletonElementOptions extends ElementOptions {}

export abstract class SkeletonElement extends Element {
  constructor(id: ElementId, options: SkeletonElementOptions) {
    super(id, options);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends SkeletonElementOptions {
  position?: Vector2;
}

export class Point extends SkeletonElement {
  static readonly defaultName = "Point";
  position: Vector2;

  constructor(id: ElementId, options: PointOptions) {
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
//                               EdgeElement

export interface EdgeElementOptions extends SkeletonElementOptions {
  startPoint: ElementId;
  endPoint: ElementId;
}

export abstract class EdgeElement extends SkeletonElement {
  startPoint: ElementId;
  endPoint: ElementId;

  constructor(id: ElementId, options: EdgeElementOptions) {
    super(id, options);
    this.startPoint = options.startPoint;
    this.endPoint = options.endPoint;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               LineSegment

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeElementOptions {}

export class LineSegment extends EdgeElement {
  static readonly defaultName = "Line Segment";

  constructor(
    readonly id: ElementId,
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

export interface ArcFromStartTangentOptions extends EdgeElementOptions {
  controlPoint?: Vector2;
}

export class ArcFromStartTangent extends EdgeElement {
  static readonly defaultName = "Arc";
  controlPoint: Vector2;

  constructor(
    readonly id: ElementId,
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

export interface CCurveOptions extends EdgeElementOptions {
  controlPoint?: Vector2;
}

export class CCurve extends EdgeElement {
  static readonly defaultName = "C-Curve";
  controlPoint: Vector2;

  constructor(
    readonly id: ElementId,
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

export interface SCurveOptions extends EdgeElementOptions {
  startControlPoint?: Vector2;
  endControlPoint?: Vector2;
}

export class SCurve extends EdgeElement {
  static readonly defaultName = "S-Curve";
  startControlPoint: Vector2;
  endControlPoint: Vector2;

  constructor(
    readonly id: ElementId,
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

export interface LayerOptions extends ElementOptions {
  elements?: Array<ElementId>;
}

export class Layer extends Element {
  static readonly defaultName = "Layer";
  elements: Array<ElementId>;

  constructor(id: ElementId, options: LayerOptions) {
    super(id, options);
    this.elements = options.elements ? [...options.elements] : [];
  }

  clone() {
    return new Layer(this.id, this);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneElementMap(source: Map<ElementId, Element>) {
  const dest = new Map<ElementId, Element>();
  source.forEach((element, id) => {
    dest.set(id, element.clone());
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
  private _elements: Map<ElementId, Element>;

  public layers: Array<ElementId>;

  constructor(other?: Document) {
    if (other) {
      this._elements = cloneElementMap(other._elements);
      this.layers = [...other.layers];
    } else {
      this._elements = new Map();
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
   * Returns the element that has the given `id`, if any:
   *
   * ```
   * const element = doc.getElementFromId(id);
   * ```
   *
   * If `type` is given as argument to this function, then this function
   * checks at runtime via `instanceof` that the element is of the given
   * `type`, and otherwise returns `undefined`:
   *
   * ```
   * const point = doc.createElement(Point);
   * const segment = doc.createElement(LineSegment);
   * const p1 = doc.getElement(point.id, Point);   // === point
   * const p2 = doc.getElement(segment.id, Point); // === undefined
   * ```
   *
   * If no `type` is given as argument to this function, but an explicit type
   * argument `T` is provided, then this function assumes the element is
   * indeed of type `T` and performs type narrowing from `Element |
   * undefined` to `T | undefined` without runtime checks. This is unsafe and
   * therefore not recommended.
   *
   * ```
   * const point = doc.createElement(Point);
   * const segment = doc.createElement(LineSegment);
   * const p1 = doc.getElement<Point>(point.id, Point);   // === point
   * const p2 = doc.getElement<Point>(segment.id, Point); // === segment as Point (bug!)
   * ```
   */
  getElement<T extends Element>(
    id: ElementId | undefined,
    type?: AbstractElementType<T> | undefined,
  ): T | undefined {
    if (!id) {
      return undefined;
    }
    const element: Element | undefined = this._elements.get(id);
    if (type === undefined) {
      // unchecked type narrowing
      return element as T | undefined;
    } else if (element instanceof type) {
      // checked type narrowing
      return element;
    } else {
      return undefined;
    }
  }

  /**
   * Returns an array of elements corresponding to the given array of `ids`.
   */
  getElements<T extends Element>(
    ids: Array<ElementId>,
    type?: AbstractElementType<T>,
  ): Array<T> {
    const res: Array<T> = [];
    for (const id of ids) {
      const element = this.getElement(id, type);
      if (element) {
        res.push(element);
      }
    }
    return res;
  }

  /**
   * Creates and returns a new element of the given `spec` with the given `options`.
   *
   * Example:
   *
   * ```
   * const p = doc.createElement(Point, {
   *   name: "New Point",
   *   position: new Vector2(42, 12),
   * });
   * ```
   *
   * Note: this does not add the element to any layer. You typically want to:
   * 1. Mutate the `elements` attribute of some layer after calling this function, or
   * 2. Use `createElementInLayer()` instead of this function.
   */
  createElement<T extends Element, Options>(
    type: ElementType<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    const element = new type(id, options);
    this._elements.set(id, element);
    return element;
  }

  foo() {
    this.createElement(Point, { name: "My Point" });
  }
  /**
   * Creates a new element of the given `spec` with the given `options`, adds it
   * as last element of the given layer, then returns it.
   *
   * If no name is provided in the options, then this function will
   * automatically a unique name withing the `layer` suitable for the given
   * `spec`, e.g., "Point 42".
   */
  createElementInLayer<T extends Element, Options extends ElementOptions>(
    type: ElementType<T, Options>,
    layer: Layer,
    options: Options,
  ): T {
    if (options.name === undefined) {
      options.name = this.findAvailableName(
        type.defaultName + " ",
        layer.elements,
      );
    }
    const element = this.createElement(type, options);
    layer.elements.push(element.id);
    return element;
  }

  /**
   * Removes the element that has the given `id` from this document.
   *
   * Note: this does not remove the element from any layer. You typically want to:
   * 1. Mutate the `elements` attribute of some layer before calling this function, or
   * 2. Use `removeElementInLayer()` instead of this function.
   */
  removeElement(id: ElementId) {
    this._elements.delete(id);
  }

  /**
   * Removes the element that has the given `id` from this document and from
   * the given `layer`.
   *
   * Note: if the element belongs to another layer than the given `layer`,
   * then it will not be removed from that other layer, which will then still
   * reference the now-stale element.
   */
  removeElementInLayer(id: ElementId, layer: Layer) {
    for (let i = layer.elements.length - 1; i >= 0; i--) {
      if (layer.elements[i] === id) {
        layer.elements.splice(i, 1);
      }
    }
    this.removeElement(id);
  }

  // TODO: Safer / more convenient API that prevents layers having stale
  // elements? For example, `layerId` could be a (readonly?) attribute of
  // each element, enforcing that each element only belongs to one layer, and
  // making it possible to automatically remove the element from its parent
  // layer in removeElement(). Also note that if instead of using the current
  // Element interface, client code were instead using some smarter Element
  // handle object that knows which document it belongs to, then it would be
  // possible to implement element.remove(), which may be an even more
  // convenient API, as close as possible as the UI equivalent of selecting
  // an element and deleting it via the delete key.

  /**
   * Finds the smallest positive integer `n` such that the name `${prefix}${n}`
   * is not taken by any of the given elements, and returns that name.
   */
  findAvailableName(prefix: string, elements: Array<ElementId>) {
    // Collect all positive integers from existing element names
    // that are of the form `${prefix}${number}`. Note that we need
    // the regex and not just rely on parseInt, since the latter
    // accepts +/-/e characters and stops at extra non-digit characters.
    const re = new RegExp(`${prefix}\\d+`);
    const numbers = [];
    for (const id of elements) {
      const element = this.getElement(id);
      if (element && re.test(element.name)) {
        const suffix = element.name.substring(prefix.length);
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
    const layer = this.createElement(Layer, { name: name });
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
    const layer = this.getElement(id, Layer);
    if (layer === undefined) {
      return this;
    }
    this.layers.splice(index, 1);
    this._elements.delete(id);
    return this;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                            Test Document

export function createTestDocument() {
  const doc = new Document();
  const layer = doc.createElement(Layer, { name: "Layer 1" });
  doc.layers = [layer.id];
  const p1 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(-100, 0),
  });
  const p2 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(0, 0),
  });
  const p3 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(100, 100),
  });
  const p4 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(200, 100),
  });
  const p5 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(200, 0),
  });
  const p6 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(100, -100),
  });
  const p7 = doc.createElementInLayer(Point, layer, {
    position: new Vector2(-100, -100),
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p1.id,
    endPoint: p2.id,
  });
  doc.createElementInLayer(ArcFromStartTangent, layer, {
    startPoint: p2.id,
    endPoint: p3.id,
    controlPoint: new Vector2(50, 0),
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p3.id,
    endPoint: p4.id,
  });
  doc.createElementInLayer(CCurve, layer, {
    startPoint: p4.id,
    endPoint: p5.id,
    controlPoint: new Vector2(150, 50),
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p5.id,
    endPoint: p6.id,
  });
  doc.createElementInLayer(SCurve, layer, {
    startPoint: p6.id,
    endPoint: p7.id,
    startControlPoint: new Vector2(50, -150),
    endControlPoint: new Vector2(-80, -60),
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p7.id,
    endPoint: p1.id,
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p2.id,
    endPoint: p6.id,
  });
  return doc;
}
