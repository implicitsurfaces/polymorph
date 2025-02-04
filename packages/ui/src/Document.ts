import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type ElementId = string;

export interface ElementBaseOptions {
  name?: string;
}

export interface ElementBaseData {
  name: string;
}

export interface ElementBase extends ElementBaseData {
  readonly id: ElementId;
  readonly type: string;
}

export interface ElementSpec<T, Options extends ElementBaseOptions> {
  readonly type: string;
  readonly name: string;
  readonly create: (id: ElementId, options: Options) => T;
  readonly clone: (other: T) => T;
}

// Note: we currently need clone() for deep cloning purposes.
//
// If all data properties of elements were immutable types (which is currently
// not the case, e.g., Vector2 and Array<ElementId>), then we could instead
// do a shallow copy via `clonedElement = {...element}` which would make
// element types more convenient to implement, and would likely improve
// performance.

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends ElementBaseOptions {
  position?: Vector2;
}

export interface PointData extends ElementBaseData {
  position: Vector2;
}

// TODO:
/*
export interface PointData extends ElementBaseData {
  x: ParamId;
  y: ParamId;
}
*/

export interface Point extends ElementBase, PointData {
  type: "Point";
}

export const Point: ElementSpec<Point, PointOptions> = {
  type: "Point",
  name: "Point",
  create: (id: ElementId, options: PointOptions) => {
    return {
      id: id,
      type: "Point",
      name: "Point",
      position: new Vector2(0, 0),
      ...options,
    };
  },
  clone: (other: Point) => {
    return { ...other, position: other.position.clone() };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               EdgeBase

export interface EdgeBaseOptions extends ElementBaseOptions {
  startPoint?: ElementId;
  endPoint?: ElementId;
}

export interface EdgeBaseData extends ElementBaseData {
  startPoint: ElementId;
  endPoint: ElementId;
}

export function isEdgeElement(element: Element): element is EdgeElement {
  return "startPoint" in element && "endPoint" in element;
}

///////////////////////////////////////////////////////////////////////////////
//                               LineSegment

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeBaseOptions {
  // No additional options w.r.t EdgeBase
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentData extends EdgeBaseData {
  // No additional data w.r.t EdgeBase
}

export interface LineSegment extends ElementBase, LineSegmentData {
  type: "LineSegment";
}

export const LineSegment: ElementSpec<LineSegment, LineSegmentOptions> = {
  type: "LineSegment",
  name: "Line Segment",
  create: (id: ElementId, options: LineSegmentOptions) => {
    return {
      id: id,
      type: "LineSegment",
      name: "Line Segment",
      startPoint: "",
      endPoint: "",
      ...options,
    };
  },
  clone: (other: LineSegment) => {
    return { ...other };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               ArcFromStartTangent

export interface ArcFromStartTangentOptions extends EdgeBaseOptions {
  tangent?: Vector2;
}

export interface ArcFromStartTangentData extends EdgeBaseData {
  tangent: Vector2;
}

export interface ArcFromStartTangent
  extends ElementBase,
    ArcFromStartTangentData {
  type: "ArcFromStartTangent";
}

export const ArcFromStartTangent: ElementSpec<
  ArcFromStartTangent,
  ArcFromStartTangentOptions
> = {
  type: "ArcFromStartTangent",
  name: "Arc",
  create: (id: ElementId, options: ArcFromStartTangentOptions) => {
    return {
      id: id,
      type: "ArcFromStartTangent",
      name: "Arc",
      startPoint: "",
      endPoint: "",
      tangent: new Vector2(1, 0),
      ...options,
    };
  },
  clone: (other: ArcFromStartTangent) => {
    return { ...other, tangent: other.tangent.clone() };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                                  CCurve

type CCurveMode = "startTangent" | "endTangent" | "controlPoint";

export interface CCurveOptions extends EdgeBaseOptions {
  controlPoint?: Vector2;
  mode?: CCurveMode;
}

export interface CCurveData extends EdgeBaseData {
  controlPoint: Vector2;
  mode: CCurveMode;
}

export interface CCurve extends ElementBase, CCurveData {
  type: "CCurve";
}

export const CCurve: ElementSpec<CCurve, CCurveOptions> = {
  type: "CCurve",
  name: "C-Curve",
  create: (id: ElementId, options: CCurveOptions) => {
    return {
      id: id,
      type: "CCurve",
      name: "C-Curve",
      startPoint: "",
      endPoint: "",
      controlPoint: new Vector2(1, 0),
      mode: "startTangent" as const,
      ...options,
    };
  },
  clone: (other: CCurve) => {
    return { ...other, controlPoint: other.controlPoint.clone() };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                                  SCurve

type SCurveMode = "tangent" | "controlPoint";

export interface SCurveOptions extends EdgeBaseOptions {
  startControlPoint?: Vector2;
  endControlPoint?: Vector2;
  mode?: SCurveMode;
}

export interface SCurveData extends EdgeBaseData {
  startControlPoint: Vector2;
  endControlPoint: Vector2;
  mode: SCurveMode;
}

export interface SCurve extends ElementBase, SCurveData {
  type: "SCurve";
}

export const SCurve: ElementSpec<SCurve, SCurveOptions> = {
  type: "SCurve",
  name: "S-Curve",
  create: (id: ElementId, options: SCurveOptions) => {
    return {
      id: id,
      type: "SCurve",
      name: "S-Curve",
      startPoint: "",
      endPoint: "",
      startControlPoint: new Vector2(1, 0),
      endControlPoint: new Vector2(1, 0),
      mode: "tangent" as const,
      ...options,
    };
  },
  clone: (other: SCurve) => {
    return {
      ...other,
      startControlPoint: other.startControlPoint.clone(),
      endControlPoint: other.endControlPoint.clone(),
    };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends ElementBaseOptions {
  elements?: Array<ElementId>;
}

export interface LayerData extends ElementBaseData {
  elements: Array<ElementId>;
}

export interface Layer extends ElementBase, LayerData {
  type: "Layer";
}

export const Layer: ElementSpec<Layer, LayerOptions> = {
  type: "Layer",
  name: "Layer",
  create: (id: ElementId, options: LayerOptions) => {
    return {
      id: id,
      type: "Layer",
      name: "Layer",
      elements: [],
      ...options,
    };
  },
  clone: (other: Layer) => {
    return { ...other, elements: [...other.elements] };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Tagged Union

export type EdgeElement = LineSegment | ArcFromStartTangent | CCurve | SCurve;
export type SkeletonElement = Point | EdgeElement;

export type Element = SkeletonElement | Layer;

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneElement(element: Element): Element {
  // Note: this could be simplified to: `return {.. element}` if all element
  // properties where immutable.
  switch (element.type) {
    case "Point":
      return Point.clone(element);
    case "Layer":
      return Layer.clone(element);
    case "LineSegment":
      return LineSegment.clone(element);
    case "ArcFromStartTangent":
      return ArcFromStartTangent.clone(element);
    case "CCurve":
      return CCurve.clone(element);
    case "SCurve":
      return SCurve.clone(element);
  }
}

function cloneElementMap(source: Map<ElementId, Element>) {
  const dest = new Map<ElementId, Element>();
  source.forEach((element, id) => {
    dest.set(id, cloneElement(element));
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
   * Returns the element that has the given `id`, if any.
   */
  getElementFromId<T extends Element>(
    id: ElementId | undefined,
  ): T | undefined {
    if (!id) {
      return undefined;
    }
    const element: Element | undefined = this._elements.get(id);
    return element as T | undefined;
  }

  /**
   * Returns an array of elements corresponding to the given array of `ids`.
   */
  getElementsFromId<T extends Element>(ids: Array<ElementId>): Array<T> {
    const res: Array<T> = [];
    for (const id of ids) {
      const element = this.getElementFromId<T>(id);
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
  createElement<T extends Element, Options extends ElementBaseOptions>(
    spec: ElementSpec<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    const element = spec.create(id, options);
    this._elements.set(id, element);
    return element;
  }

  /**
   * Creates a new element of the given `spec` with the given `options`, adds it
   * as last element of the given layer, then returns it.
   *
   * If no name is provided in the options, then this function will
   * automatically a unique name withing the `layer` suitable for the given
   * `spec`, e.g., "Point 42".
   */
  createElementInLayer<T extends Element, Options extends ElementBaseOptions>(
    spec: ElementSpec<T, Options>,
    layer: Layer,
    options: Options,
  ): T {
    if (options.name === undefined) {
      options.name = this.findAvailableName(spec.name + " ", layer.elements);
    }
    const element = this.createElement(spec, options);
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
      const element = this.getElementFromId(id);
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
    const layer = this.getElementFromId<Layer>(id);
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
    tangent: new Vector2(50, 0),
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p3.id,
    endPoint: p4.id,
  });
  doc.createElementInLayer(CCurve, layer, {
    startPoint: p4.id,
    endPoint: p5.id,
    controlPoint: new Vector2(-50, -50),
    mode: "startTangent",
  });
  doc.createElementInLayer(LineSegment, layer, {
    startPoint: p5.id,
    endPoint: p6.id,
  });
  doc.createElementInLayer(SCurve, layer, {
    startPoint: p6.id,
    endPoint: p7.id,
    startControlPoint: new Vector2(-50, -50),
    endControlPoint: new Vector2(20, 40),
    mode: "tangent",
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
