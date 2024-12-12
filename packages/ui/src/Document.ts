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

export interface ElementSpec<T, Options> {
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

export interface Point extends ElementBase, PointData {
  type: "Point";
}

export const PointDefaultOptions = {
  name: "Point",
  position: new Vector2(0, 0),
};

export const Point: ElementSpec<Point, PointOptions> = {
  create: (id: ElementId, options: PointOptions) => {
    return {
      id: id,
      type: "Point",
      ...PointDefaultOptions,
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

export const LineSegmentDefaultOptions = {
  name: "Line Segment",
  startPoint: "",
  endPoint: "",
};

export const LineSegment: ElementSpec<LineSegment, LineSegmentOptions> = {
  create: (id: ElementId, options: LineSegmentOptions) => {
    return {
      id: id,
      type: "LineSegment",
      ...LineSegmentDefaultOptions,
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

export const ArcFromStartTangentDefaultOptions = {
  name: "Arc",
  startPoint: "",
  endPoint: "",
  tangent: new Vector2(1, 0),
};

export const ArcFromStartTangent: ElementSpec<
  ArcFromStartTangent,
  ArcFromStartTangentOptions
> = {
  create: (id: ElementId, options: ArcFromStartTangentOptions) => {
    return {
      id: id,
      type: "ArcFromStartTangent",
      ...ArcFromStartTangentDefaultOptions,
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

export const CCurveDefaultOptions = {
  name: "C-Curve",
  startPoint: "",
  endPoint: "",
  controlPoint: new Vector2(1, 0),
  mode: "startTangent" as const,
};

export const CCurve: ElementSpec<CCurve, CCurveOptions> = {
  create: (id: ElementId, options: CCurveOptions) => {
    return {
      id: id,
      type: "CCurve",
      ...CCurveDefaultOptions,
      ...options,
    };
  },
  clone: (other: CCurve) => {
    return { ...other, controlPoint: other.controlPoint.clone() };
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

export const LayerDefaultOptions = {
  name: "Layer",
  elements: [],
};

export const Layer: ElementSpec<Layer, LayerOptions> = {
  create: (id: ElementId, options: LayerOptions) => {
    return {
      id: id,
      type: "Layer",
      ...LayerDefaultOptions,
      ...options,
    };
  },
  clone: (other: Layer) => {
    return { ...other, elements: [...other.elements] };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Tagged Union

export type EdgeElement = LineSegment | ArcFromStartTangent | CCurve;
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
  copy.sort();
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

  getElementFromId<T extends Element>(id: ElementId): T | undefined {
    const element: Element | undefined = this._elements.get(id);
    return element as T | undefined;
  }

  createElement<T extends Element, Options>(
    spec: ElementSpec<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    const element = spec.create(id, options);
    this._elements.set(id, element);
    return element;
  }

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
  const p1 = doc.createElement(Point, {
    name: "Point 1",
    position: new Vector2(0, 0),
  });
  const p2 = doc.createElement(Point, {
    name: "Point 2",
    position: new Vector2(100, 0),
  });
  const p3 = doc.createElement(Point, {
    name: "Point 3",
    position: new Vector2(200, 100),
  });
  const p4 = doc.createElement(Point, {
    name: "Point 4",
    position: new Vector2(200, 200),
  });
  const p5 = doc.createElement(Point, {
    name: "Point 5",
    position: new Vector2(300, 300),
  });
  const p6 = doc.createElement(Point, {
    name: "Point 6",
    position: new Vector2(400, 300),
  });
  const s1 = doc.createElement(LineSegment, {
    name: "Segment 1",
    startPoint: p1.id,
    endPoint: p2.id,
  });
  const arc = doc.createElement(ArcFromStartTangent, {
    name: "Arc 1",
    startPoint: p2.id,
    endPoint: p3.id,
    tangent: new Vector2(50, 0),
  });
  const s2 = doc.createElement(LineSegment, {
    name: "Segment 2",
    startPoint: p3.id,
    endPoint: p4.id,
  });
  const cc = doc.createElement(CCurve, {
    name: "C-Curve 1",
    startPoint: p4.id,
    endPoint: p5.id,
    controlPoint: new Vector2(0, 50),
    mode: "startTangent",
  });
  const s3 = doc.createElement(LineSegment, {
    name: "Segment 3",
    startPoint: p5.id,
    endPoint: p6.id,
  });
  layer.elements = [p1, p2, p3, p4, p5, p6, s1, arc, s2, cc, s3].map(
    (e) => e.id,
  );
  return doc;
}
